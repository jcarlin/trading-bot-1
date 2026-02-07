"""Tests for core.config module."""

import os
import tempfile
import unittest

import yaml


class TestConfig(unittest.TestCase):
    """Tests for the centralized Config class."""

    def _write_yaml(self, data: dict) -> str:
        """Write data to a temp YAML file, return path."""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(data, f)
        f.close()
        self._tmp_files.append(f.name)
        return f.name

    def setUp(self):
        self._tmp_files = []

    def tearDown(self):
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_from_yaml_basic(self):
        from core.config import Config
        path = self._write_yaml({'exchange': {'name': 'binance'}, 'symbol': 'BTC/USDT'})
        config = Config.from_yaml(path)
        self.assertEqual(config.get('exchange.name'), 'binance')
        self.assertEqual(config.get('symbol'), 'BTC/USDT')

    def test_get_with_default(self):
        from core.config import Config
        path = self._write_yaml({'a': 1})
        config = Config.from_yaml(path)
        self.assertIsNone(config.get('nonexistent'))
        self.assertEqual(config.get('nonexistent', 42), 42)

    def test_require_raises(self):
        from core.config import Config
        path = self._write_yaml({'a': 1})
        config = Config.from_yaml(path)
        with self.assertRaises(ValueError):
            config.require('missing.key')

    def test_require_returns_value(self):
        from core.config import Config
        path = self._write_yaml({'exchange': {'name': 'test'}})
        config = Config.from_yaml(path)
        self.assertEqual(config.require('exchange.name'), 'test')

    def test_get_section(self):
        from core.config import Config
        path = self._write_yaml({'risk': {'max_drawdown': 0.2, 'commission': 0.001}})
        config = Config.from_yaml(path)
        section = config.get_section('risk')
        self.assertIsInstance(section, dict)
        self.assertEqual(section['max_drawdown'], 0.2)

    def test_get_section_missing(self):
        from core.config import Config
        path = self._write_yaml({'a': 1})
        config = Config.from_yaml(path)
        self.assertEqual(config.get_section('nonexistent'), {})

    def test_env_var_substitution(self):
        from core.config import Config
        os.environ['TEST_CONFIG_VAR'] = 'my_secret'
        try:
            path = self._write_yaml({'api_key': '${TEST_CONFIG_VAR}'})
            config = Config.from_yaml(path)
            self.assertEqual(config.get('api_key'), 'my_secret')
        finally:
            del os.environ['TEST_CONFIG_VAR']

    def test_env_var_with_default(self):
        from core.config import Config
        # Ensure the env var does NOT exist
        os.environ.pop('TEST_MISSING_VAR', None)
        path = self._write_yaml({'val': '${TEST_MISSING_VAR:-fallback_value}'})
        config = Config.from_yaml(path)
        self.assertEqual(config.get('val'), 'fallback_value')

    def test_env_var_override_default(self):
        from core.config import Config
        os.environ['TEST_OVERRIDE_VAR'] = 'real_value'
        try:
            path = self._write_yaml({'val': '${TEST_OVERRIDE_VAR:-fallback}'})
            config = Config.from_yaml(path)
            self.assertEqual(config.get('val'), 'real_value')
        finally:
            del os.environ['TEST_OVERRIDE_VAR']

    def test_nested_env_substitution(self):
        from core.config import Config
        os.environ['TEST_NESTED'] = 'found'
        try:
            path = self._write_yaml({'a': {'b': {'c': '${TEST_NESTED}'}}})
            config = Config.from_yaml(path)
            self.assertEqual(config.get('a.b.c'), 'found')
        finally:
            del os.environ['TEST_NESTED']

    def test_contains(self):
        from core.config import Config
        path = self._write_yaml({'exchange': {'name': 'binance'}})
        config = Config.from_yaml(path)
        self.assertIn('exchange.name', config)
        self.assertNotIn('missing', config)

    def test_raw_property(self):
        from core.config import Config
        data = {'a': 1, 'b': {'c': 2}}
        path = self._write_yaml(data)
        config = Config.from_yaml(path)
        self.assertEqual(config.raw, data)

    def test_file_not_found(self):
        from core.config import Config
        with self.assertRaises(FileNotFoundError):
            Config.from_yaml('/nonexistent/path.yaml')

    def test_backward_compat_default_yaml(self):
        """Ensure Config can load the existing config/default.yaml without error."""
        from core.config import Config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
        if os.path.exists(config_path):
            config = Config.from_yaml(config_path)
            self.assertEqual(config.get('exchange.name'), 'binance')
            self.assertTrue(config.get('exchange.sandbox'))
            self.assertEqual(config.get('symbol'), 'BTC/USDT')
            self.assertEqual(config.get('strategy.name'), 'sma_crossover')


if __name__ == '__main__':
    unittest.main()
