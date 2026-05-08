# gateway/__init__.py — Helios API Gateway layer
# Author: Hridam Biswas | Project: Helios

from gateway.router import GatewayMiddleware

__all__ = ["GatewayMiddleware"]
