FROM postgres:16

# Update and install dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      locales \
      ca-certificates \
      build-essential \
      postgresql-server-dev-all \
      libicu-dev \
      libpam0g-dev \
      zlib1g-dev \
      git \
      curl

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install pg_vector
RUN cd /tmp && \
    git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install && \
    rm -rf /tmp/pgvector

RUN echo 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";' >> /docker-entrypoint-initdb.d/uuid-ossp.sql
RUN echo 'CREATE EXTENSION IF NOT EXISTS vector;' >> /docker-entrypoint-initdb.d/vector.sql

# Clean up
RUN apt-get remove -y --auto-remove \
      build-essential \
      postgresql-server-dev-all \
      libicu-dev \
      libpam0g-dev \
      zlib1g-dev \
      git \
      curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*