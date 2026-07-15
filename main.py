import os
import json

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)



def _as_text_list(text_samples):
  """Normalize generated text output into a Python list of strings."""
  if text_samples is None:
    return []
  if isinstance(text_samples, str):
    return [text_samples]
  return [str(x) for x in list(text_samples)]


@L.pytorch.utilities.rank_zero_only
def _save_generated_samples(config, texts, filename_prefix='generated'):
  """Save all generated samples to jsonl for external generation metrics.

  The printed `Text samples` only shows the last batch. This function saves
  every decoded sample generated across `sampling.num_sample_batches`, so the
  saved file should contain eval_batch_size * num_sample_batches examples.
  """
  output_dir = config.checkpointing.save_dir
  sample_path = os.path.join(
    output_dir,
    f'{filename_prefix}_steps{config.sampling.steps}.jsonl')

  with fsspec.open(sample_path, 'w') as fp:
    for i, text in enumerate(texts):
      fp.write(json.dumps({
        'id': i,
        'steps': int(config.sampling.steps),
        'text': text,
      }, ensure_ascii=False) + '\n')

  print(f'Saved all generated samples to: {sample_path}')
  print(f'Number of saved samples: {len(texts)}')


def compute_sample_eval_generation_metrics(
    sample_batches,
    tokenizer,
    mask_index=None):
  """Compute sample_eval diversity and repetition metrics.

  This uses raw generated token ids. It removes special tokens but does
  not stop at EOS, because MDLM samples may contain many EOS tokens.
  """

  special_ids = {
    getattr(tokenizer, 'pad_token_id', None),
    getattr(tokenizer, 'eos_token_id', None),
    getattr(tokenizer, 'bos_token_id', None),
    getattr(tokenizer, 'mask_token_id', None),
    mask_index,
  }
  special_ids = {x for x in special_ids if x is not None}

  per_sample_tokens = []
  all_tokens = []

  for batch in sample_batches:
    batch = batch.detach().cpu()

    for seq in batch.tolist():
      toks = []
      for tok in seq:
        if tok in special_ids:
          continue
        toks.append(tok)

      per_sample_tokens.append(toks)
      all_tokens.extend(toks)

  if len(all_tokens) > 0:
    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    counts = torch.bincount(token_tensor)
    probs = counts[counts > 0].float()
    probs = probs / probs.sum()
    gen_entropy = float(-(probs * probs.log()).sum().item())
  else:
    gen_entropy = 0.0

  def distinct_n(samples, n):
    total = 0
    unique = set()

    for toks in samples:
      if len(toks) < n:
        continue
      for i in range(len(toks) - n + 1):
        unique.add(tuple(toks[i: i + n]))
        total += 1

    if total == 0:
      return 0.0
    return len(unique) / total

  distinct_2 = distinct_n(per_sample_tokens, 2)
  distinct_4 = distinct_n(per_sample_tokens, 4)
  repetition_2 = 1.0 - distinct_2
  repetition_4 = 1.0 - distinct_4

  lengths = [len(toks) for toks in per_sample_tokens]
  avg_gen_len = sum(lengths) / len(lengths) if len(lengths) > 0 else 0.0

  return {
    'gen_entropy': gen_entropy,
    'distinct_2': distinct_2,
    'distinct_4': distinct_4,
    'repetition_2': repetition_2,
    'repetition_4': repetition_4,
    'avg_gen_len': avg_gen_len,
  }


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  sample_batches = []
  text_samples = None

  # Store every generated sample, not only the last printed batch.
  # With loader.eval_batch_size=1 and sampling.num_sample_batches=200,
  # this list should contain 200 strings.
  all_text_samples = []

  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      all_text_samples.extend(_as_text_list(text_samples))
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)

      sample_batches.append(samples.detach().cpu())

      # Keep the original decoded text for generative perplexity, preserving
      # the previous behavior of compute_generative_perplexity().
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)

      # Save a cleaner version for MAUVE / Self-BLEU / qualitative examples.
      # The raw token ids are still kept in sample_batches for Distinct etc.
      clean_text_samples = model.tokenizer.batch_decode(
        samples,
        skip_special_tokens=True)
      all_text_samples.extend(_as_text_list(clean_text_samples))

  print('Text samples:', text_samples)
  _save_generated_samples(config, all_text_samples)

  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())

    gen_metrics = compute_sample_eval_generation_metrics(
      sample_batches=sample_batches,
      tokenizer=model.tokenizer,
      mask_index=model.mask_index)

    print(f"Generation entropy: {gen_metrics['gen_entropy']:.4f}")
    print(f"Distinct-2: {gen_metrics['distinct_2']:.4f}")
    print(f"Distinct-4: {gen_metrics['distinct_4']:.4f}")
    print(f"Repetition-2: {gen_metrics['repetition_2']:.4f}")
    print(f"Repetition-4: {gen_metrics['repetition_4']:.4f}")
    print(f"Average generated length: {gen_metrics['avg_gen_len']:.2f}")

  return all_text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()