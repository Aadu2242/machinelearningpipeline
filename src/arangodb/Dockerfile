FROM arangodb:latest
COPY entrypoint2.sh /entrypoint2.sh
RUN chmod +x /entrypoint2.sh
ENV ARANGO_ROOT_PASSWORD_FILE /run/secrets/arangodb-root-password
ENTRYPOINT ["/entrypoint2.sh"]