try:
    from .pointrope_cuda import PointROPE
except Exception as e:
    print(
        f"[PointROPE] CUDA implementation unavailable ({type(e).__name__}: {e}). "
        "Using slower Pytorch fallback."
    )
    from .pointrope_torch import PointROPE

__all__ = ["PointROPE"]
