from logging import getLogger
import huggingface_hub as hfhub


logger = getLogger(__name__)


class PathProvider:
    """A wrapper around folder_paths.get_folder_paths."""

    def get_folder_paths(self, *args, **kwargs):
        import folder_paths

        return folder_paths.get_folder_paths(*args, **kwargs)


class ComfyLoader:
    """A wrapper around comfy.sd.load_checkpoint_guess_config."""

    def load_checkpoint(self, *args, **kwargs):
        import comfy.sd

        return comfy.sd.load_checkpoint_guess_config(*args, **kwargs)


class ModelDownloader:
    """A node that downloads a model from Hugging Face Hub before running the execute method."""

    def __init__(self):
        # TODO: get token from server
        # hfhub.login(token=...)
        pass

    @staticmethod
    def download_model(model_name: str, revision: str):
        """Download a model from Hugging Face Hub."""
        try:
            path = hfhub.hf_hub_download(
                repo_id=model_name,
                revision=revision,
            )
        except hfhub.hf_api.HfHubHTTPError as e:
            logger.exception("Error downloading model %s. Response: %s", model_name, e)
            e.add_note(f"Model {model_name} could not be downloaded.")
            raise ValueError("Error downloading model") from e
        return path


class HfCheckpointLoader:
    """A node that downloads a model from Hugging Face Hub before running the execute method."""

    def __init__(self):
        # TODO: get token from server
        self.path_provider = PathProvider()
        self.comfy_loader = ComfyLoader()
        self.model_downloader = ModelDownloader()
        # hfhub.login(token=...)
        pass

    FUNCTION = "load_checkpoint"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING",),
                "revision": ("STRING", {"default": "main"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
    )

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint from HuggingFace, diffusion models are used to denoise latents."

    def load_checkpoint(self, model_name: str):
        ckpt_path = self.model_downloader.download_model(model_name, "main")
        out = self.comfy_loader.load_checkpoint(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=self.path_provider.get_folder_paths("embeddings"),
        )
        return out[:3]


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"HfCheckpointLoader": HfCheckpointLoader}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"HfCheckpointLoader": "HuggingFace Checkpoint Loader"}
