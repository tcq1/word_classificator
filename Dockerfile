ARG BASE_IMAGE=python:3.10.8-slim-buster
FROM $BASE_IMAGE

# install poetry
RUN pip install poetry==1.2.2

# install project requirements
COPY pyproject.toml poetry.lock /
RUN poetry config virtualenvs.create false && poetry install

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

# copy the whole project except what is in .dockerignore
WORKDIR /home/kedro
COPY . .
RUN chown -R kedro:${KEDRO_GID} /home/kedro
USER kedro
RUN chmod -R a+w /home/kedro

EXPOSE 8888

CMD ["kedro", "run"]
