# MOSS-Image support for Forge Neo
# The loader check of compat.py reads Forge's own loader source, even after another extension put a wrapper on
# load_huggingface_component (issue #1: the FLUX 3 extension loads first and does exactly that).

import importlib.util
import inspect
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

from forge_moss.native import compat

LOADER = '''
def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    if cls_name == "AutoencoderKLFlux2":
        pass
    if cls_name in ["Qwen3Model", "Qwen3VLModel"]:
        from backend.nn.llm.llama import Qwen3_8B
    elif cls_name == "Krea2Transformer2DModel":
        from backend.nn.krea import SingleStreamDiT
'''

# what the FLUX 3 extension's hook looks like: its own branches, and the original for everything else
HOOK = '''
def hook(loader):
    original = loader.load_huggingface_component

    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if cls_name == "AutoencoderKLFlux2":
            pass
        return original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component
'''


def operations(add=True):
    def init(self, *args, add=False, **kwargs):
        pass

    def init_without(self, *args, **kwargs):
        pass

    norm = type("RMSNorm", (), {"__init__": init if add else init_without})
    return types.SimpleNamespace(ForgeOperations=types.SimpleNamespace(RMSNorm=norm))


class TestLoaderProblems(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def load(self, name, source):
        path = Path(self.directory.name) / f"{name}.py"
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def loader(self, source):
        return self.load("fake_forge_loader", source)

    def test_forge_loader(self):
        self.assertEqual(compat.loader_problems(self.loader(LOADER), operations()), [])

    def test_wrapped_by_another_extension(self):
        module = self.loader(LOADER)
        self.load("fake_extension", HOOK).hook(module)
        # the function's own source is now the hook's, which is what the check used to read
        self.assertNotIn("Qwen3Model", inspect.getsource(module.load_huggingface_component))
        self.assertEqual(compat.loader_problems(module, operations()), [])

    def test_loader_without_the_qwen3_branch(self):
        module = self.loader(LOADER.replace('"Qwen3Model", ', ""))
        self.assertEqual(compat.loader_problems(module, operations()), ["the model loader has no Qwen3Model branch"])

    def test_loader_that_imports_elsewhere(self):
        module = self.loader(LOADER.replace("from backend.nn.krea import SingleStreamDiT", "pass"))
        self.assertEqual(compat.loader_problems(module, operations()),
                         ["the model loader no longer does from backend.nn.krea import SingleStreamDiT"])

    def test_rmsnorm_without_add(self):
        self.assertEqual(compat.loader_problems(self.loader(LOADER), operations(add=False)),
                         ["backend.operations.RMSNorm has no add argument"])


if __name__ == "__main__":
    unittest.main()
