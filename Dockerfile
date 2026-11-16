FROM python:3.13-slim
LABEL authors="ana"

# Instalamos Poetry en el contenedor, ya que solo tiene instalado Python
ENV POETRY_VERSION=2.2.1
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

# Copiar archivos de dependencias, aplicación y modelos
COPY . .

# Configurar Poetry para instalar en el entorno del contenedor
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

EXPOSE 7860

CMD ["python", "src/taller_ii_boilerplate/app.py"]