"""Thin extension router; all market/ledger work lives in the service."""

from fastapi import APIRouter, HTTPException, Request

from app.extensions import BACKEND_EXTENSION_API_VERSION, BackendExtensionRegistrar
from app.services.t_strategy import ManualRequest, TConfig, TStrategyService

EXTENSION_ID = "official.t-strategy"
EXTENSION_API_VERSION = BACKEND_EXTENSION_API_VERSION
router = APIRouter(prefix="/api/t-strategy", tags=["t-strategy"])


def _service(request: Request) -> TStrategyService:
    repo = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(503, "本地数据仓库尚未就绪")
    return TStrategyService(repo)


def _call(action, *args):
    try:
        return action(*args)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/status")
def status(request: Request):
    return _call(_service(request).status)


@router.post("/backtest")
def backtest(request: Request, body: TConfig):
    return _call(_service(request).backtest, body)


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: str):
    return _call(_service(request).get_run, run_id)


@router.post("/runs/{run_id}/paper")
def import_paper(request: Request, run_id: str):
    return _call(_service(request).import_paper, run_id)


@router.post("/manual/step")
def manual(request: Request, body: ManualRequest):
    return _call(_service(request).manual_step, body)


def setup(registrar: BackendExtensionRegistrar) -> None:
    registrar.include_router(router)
