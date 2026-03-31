from __future__ import annotations

import importlib
import sys
from unittest import TestCase, main


class CLIHardeningTest(TestCase):
    def test_cli_import_keeps_heavy_scenarios_lazy(self) -> None:
        heavy_modules = [
            "rdagent.app.data_science.loop",
            "rdagent.app.finetune.llm.loop",
            "rdagent.app.general_model.general_model",
            "rdagent.app.qlib_rd_loop.factor",
            "rdagent.app.qlib_rd_loop.factor_from_report",
            "rdagent.app.qlib_rd_loop.model",
            "rdagent.app.qlib_rd_loop.quant",
            "rdagent.log.mle_summary",
        ]

        sys.modules.pop("rdagent.app.cli", None)
        for module_name in heavy_modules:
            sys.modules.pop(module_name, None)

        cli_module = importlib.import_module("rdagent.app.cli")

        self.assertTrue(callable(cli_module.runtime_info_cli))
        self.assertTrue(callable(cli_module.health_check_cli))
        for module_name in heavy_modules:
            self.assertNotIn(module_name, sys.modules)


if __name__ == "__main__":
    main()
