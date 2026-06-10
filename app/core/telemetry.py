"""
Configuracion de OpenTelemetry para los servicios FastAPI.

Reutilizable por credit-ai-service y vault-mock. Llamar a setup_telemetry(app, "<nombre>")
una vez, despues de crear la app FastAPI.

El FastAPIInstrumentor extrae automaticamente el header W3C 'traceparent' que envia
el orquestador (Spring Boot), de modo que los spans de este servicio se enganchan
como hijos de la traza distribuida ya existente.
"""
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_telemetry(app, service_name: str) -> None:
    # El nombre que aparecera en la UI de Jaeger.
    resource = Resource.create({SERVICE_NAME: service_name})

    provider = TracerProvider(resource=resource)

    # Para apuntar a un colector gestionado (AWS X-Ray / Azure Monitor) en produccion,
    # basta con definir la variable de entorno OTEL_EXPORTER_OTLP_TRACES_ENDPOINT.
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://localhost:4318/v1/traces",
    )

    # BatchSpanProcessor exporta en segundo plano: overhead despreciable por request.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    # Instrumenta requests entrantes + extrae el contexto de traza propagado.
    FastAPIInstrumentor.instrument_app(app)