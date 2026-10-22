"""base 설치(extras 없음)에서 패키지 import 가 깨지지 않는지.

extras 가 다 깔린 개발 환경에서는 ImportError 안내 테스트가 skip 된다.
`uv sync` (extras 없이) 한 환경에서 이 파일만 돌려야 의미가 있다.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from playground_corpus import write_documents

EXTRAS_MODULES = ("pyarrow", "torch", "sentence_transformers")


def test_import_does_not_load_extras_modules():
    # 새 프로세스에서 import 해야 다른 테스트가 먼저 올린 모듈의 영향을 받지 않는다
    code = (
        "import sys, playground_corpus; "
        f"print(','.join(m for m in {EXTRAS_MODULES!r} if m in sys.modules))"
    )
    result = subprocess.run(  # noqa: S603  # 고정 인자로 현재 인터프리터 실행
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == ""


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is not None, reason="pyarrow 가 설치된 환경"
)
def test_parquet_functions_explain_missing_extras(tmp_path: Path):
    with pytest.raises(ImportError, match=r"playground-corpus\[parquet\]"):
        write_documents([], tmp_path / "docs.parquet")
