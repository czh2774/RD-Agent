from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from rdagent.utils.env import QlibCondaConf, QlibCondaEnv


class QlibCondaHardeningTest(TestCase):
    def test_qlib_conda_conf_uses_editable_install_for_local_source(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            qlib_source = Path(tmp_dir) / "qlib"
            qlib_source.mkdir()

            conf = QlibCondaConf(qlib_pip_install_target=str(qlib_source))

            self.assertTrue(conf.should_use_editable_install())
            self.assertIn("pip install -e ", conf.build_qlib_pip_install_command())
            self.assertIn(str(qlib_source), conf.build_qlib_pip_install_command())

    def test_qlib_conda_env_prepare_uses_configured_install_target(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            qlib_source = Path(tmp_dir) / "qlib"
            qlib_source.mkdir()

            env = QlibCondaEnv(
                conf=QlibCondaConf(
                    qlib_pip_install_target=str(qlib_source),
                    bootstrap_packages="catboost tables",
                )
            )

            with (
                patch.object(env, "_env_exists", return_value=False),
                patch.object(env, "_qlib_import_available", return_value=False),
                patch("subprocess.check_call") as mock_check_call,
            ):
                env.prepare()

            commands = [call.args[0] for call in mock_check_call.call_args_list]
            self.assertTrue(any("conda create -y -n" in command for command in commands))
            self.assertTrue(any("pip install -e" in command and str(qlib_source) in command for command in commands))
            self.assertFalse(any("git+https://github.com/microsoft/qlib.git" in command for command in commands))
            self.assertTrue(any("pip install catboost tables" in command for command in commands))

    def test_qlib_conda_env_prepare_skips_bootstrap_when_env_is_ready(self) -> None:
        env = QlibCondaEnv(conf=QlibCondaConf())

        with (
            patch.object(env, "_env_exists", return_value=True),
            patch.object(env, "_qlib_import_available", return_value=True),
            patch("subprocess.check_call") as mock_check_call,
        ):
            env.prepare()

        mock_check_call.assert_not_called()


if __name__ == "__main__":
    main()
