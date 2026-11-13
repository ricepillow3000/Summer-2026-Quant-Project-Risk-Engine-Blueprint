# Meliona in a box: the same app on any machine that has Docker.
#
#   docker run -p 8501:8501 ghcr.io/ricepillow3000/meliona:latest      # prebuilt
#   docker build -f docker/Dockerfile -t meliona .                      # or build it
#   docker run -p 8501:8501 meliona
#
# then open http://localhost:8501. The box runs exactly what the Procfile runs
# on Railway (test_the_container_runs_what_the_procfile_runs), so "it works on
# my machine" and "it works on the live site" are one claim.
#
# Not at the repo root on purpose: Railway builds a root Dockerfile instead of
# its own builder, and the live site's build must not change by accident.
# Pinned to one build of the base image: a tag can be moved to new code, a
# digest cannot. Dependabot proposes the next digest (.github/dependabot.yml).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PORT=8501
WORKDIR /app

# Dependencies before code: a code change reuses this layer, not a reinstall.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# The served page's head (title, link preview, boot skeleton, reveal.js) is
# written into Streamlit's own index.html. Railway does it at every start; the
# box does it once, here, while those files still belong to root.
ARG BUILD_SHA=""
RUN RAILWAY_GIT_COMMIT_SHA="$BUILD_SHA" python -m tools.public_files

# Not root: a hole in the app would not own the box. data/ (the price cache)
# and logs/ (the crash log) are the only places it writes.
RUN useradd --create-home --uid 10001 meliona \
 && mkdir -p data logs && chown meliona data logs
USER meliona

EXPOSE 8501
# Docker marks the box unhealthy after three failed checks, and anything that
# watches it (Railway, a restart policy, an orchestrator) can act on that.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('PORT', '8501'), timeout=4)"

# As the Procfile: the cache warmer beside the app, then the app.
CMD ["sh", "-c", "(python -m tools.warm_cache --every 5 &) && exec streamlit run serve.py --server.port=${PORT} --server.address=0.0.0.0"]
