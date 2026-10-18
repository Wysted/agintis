# Postgres for agintis: pgvector + pg_textsearch (BM25, PostgreSQL license, Timescale).
# Used by the `db` service of docker-compose.yml and by the CI integration job (built the same way).
# Both stages MUST use the same base image: the extension is built against that server's headers.

FROM pgvector/pgvector:0.8.7-pg17 AS build
# The tag is pinned AND checked against its commit (a moved tag fails the build).
ARG PG_TEXTSEARCH_TAG=v1.5.1
ARG PG_TEXTSEARCH_COMMIT=fa142d049d08eac45a263f5abc250c8877ae076e
# Headers pinned to the base server's exact package version (PG_VERSION comes from the base image)
# for an exact ABI match. Once PGDG moves that minor to apt-archive the pin cannot resolve: fall
# back to the current headers (minor releases keep the ABI except in rare, announced cases).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates git \
 && { apt-get install -y --no-install-recommends "postgresql-server-dev-${PG_MAJOR}=${PG_VERSION}" \
      || apt-get install -y --no-install-recommends "postgresql-server-dev-${PG_MAJOR}"; } \
 && git -c advice.detachedHead=false clone --depth 1 --branch "${PG_TEXTSEARCH_TAG}" \
    https://github.com/timescale/pg_textsearch /tmp/pg_textsearch \
 && test "$(git -C /tmp/pg_textsearch rev-parse HEAD)" = "${PG_TEXTSEARCH_COMMIT}" \
 && make -C /tmp/pg_textsearch \
 && make -C /tmp/pg_textsearch install DESTDIR=/out

# Runtime: the base image plus the extension files only (no compilers).
FROM pgvector/pgvector:0.8.7-pg17
COPY --from=build /out/ /
# pg_textsearch must be preloaded. Set here (not in postgresql.conf) so it also applies to
# existing data volumes and to CI, where `services:` containers cannot pass a command.
CMD ["postgres", "-c", "shared_preload_libraries=pg_textsearch"]
