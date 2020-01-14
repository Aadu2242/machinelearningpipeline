#!/bin/sh
export ARANGO_ROOT_PASSWORD=`cat ${test}`
/entrypoint.sh "$@"