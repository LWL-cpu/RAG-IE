import torch
from typing import TYPE_CHECKING, Optional, Set, Tuple, List
from peft import PeftModel, TaskType, LoraConfig, get_peft_model

# from utils.logging import get_logger
# from utils.constants import LAYERNORM_NAMES
# from args import FinetuningArguments, ModelArguments


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

# logger = get_logger(__name__)



# def find_all_linear_modules(
#     model: "PreTrainedModel",
#     quantization_bit: Optional[int] = None
# ) -> List[str]:
#     r"""
#     Finds all available modules to apply lora.
#     """
#     if quantization_bit is not None:
#         import bitsandbytes as bnb
#         linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
#     else:
#         linear_cls = torch.nn.Linear
#
#     output_layer_names = ["lm_head"]
#     if model.config.model_type == "chatglm":
#         output_layer_names.append("output_layer")
#
#     module_names = set()
#     for name, module in model.named_modules():
#         if (
#             isinstance(module, linear_cls)
#             and not any([output_layer in name for output_layer in output_layer_names])
#         ):
#             module_names.add(name.split(".")[-1])
#
#     logger.info("Found linear modules: {}".format(",".join(module_names)))
#     return list(module_names)






def init_adapter(
    model: "PreTrainedModel",
    is_trainable: bool,

) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    # if (not is_trainable) and model_args.checkpoint_dir is None:
    #     logger.info("Checkpoint is not found at evaluation, load the original model.")
    #     return model
    #
    # if finetuning_args.finetuning_type == "full" and is_trainable:
    #     logger.info("Fine-tuning method: Full")
    #     model = model.float()
    #
    # if finetuning_args.finetuning_type == "freeze" and is_trainable:
    #     logger.info("Fine-tuning method: Freeze")
    #     num_layers = (
    #         getattr(model.config, "num_hidden_layers", None)
    #         or getattr(model.config, "num_layers", None)
    #         or getattr(model.config, "n_layer", None)
    #     )
    #     if not num_layers:
    #         raise ValueError("Current model does not support freeze tuning.")
    #     if finetuning_args.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
    #         trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
    #     else: # fine-tuning the first n layers if num_layer_trainable < 0
    #         trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]
    #
    #     trainable_layers = []
    #     for module_name in finetuning_args.name_module_trainable:
    #         for idx in trainable_layer_ids:
    #             trainable_layers.append("{:d}.{}".format(idx, module_name))
    #
    #     for name, param in model.named_parameters():
    #         if not any(trainable_layer in name for trainable_layer in trainable_layers):
    #             param.requires_grad_(False)
    #         else:
    #             param.data = param.data.to(torch.float32)

    # if finetuning_args.finetuning_type == "lora":
    if True:
        print("Fine-tuning method: LoRA")
        checkpoint_to_resume = None
        resume_lora_training = True

        # if checkpoint_dir is not None:
        #     if is_trainable and resume_lora_training:
        #         checkpoints_to_merge = checkpoint_dir
        #     else:
        #         checkpoints_to_merge = checkpoint_dir
        #
        #     for checkpoint in [checkpoints_to_merge]:
        #         model = PeftModel.from_pretrained(model, checkpoint)
        #         model = model.merge_and_unload()

            # if len(checkpoints_to_merge) > 0:
            #     logger.info("Merged model checkpoint(s): {}.".format(checkpoints_to_merge))
            #
            # if checkpoint_to_resume is not None: # resume lora training
            #     logger.info("Resume model checkpoint(s): {} .".format(checkpoint_to_resume))
            #     model = PeftModel.from_pretrained(model, checkpoint_to_resume, is_trainable=is_trainable)

        if is_trainable and checkpoint_to_resume is None:    # create new lora weights while training
            # if len(finetuning_args.lora_target_modules) == 1 and finetuning_args.lora_target_modules[0] == "all":
            #     target_modules = find_all_linear_modules(model, model_args.bits)
            # else:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'gate']

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=target_modules,
                modules_to_save=["gate"]
            )
            model = get_peft_model(model, lora_config)

    # if model_args.checkpoint_dir is not None:
    #     logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model