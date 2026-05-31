FROM hmcvlab/computer-vision:latest
USER root
# act copies helpers into /var/run/act, but /var/run is a symlink to /run
# in this image, which docker cp refuses to traverse. Replace with a real dir.
RUN rm -f /var/run && mkdir -p /var/run && ln -s /var/run /run-compat || true
