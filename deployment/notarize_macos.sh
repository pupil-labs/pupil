#!/bin/bash

# Based on
# https://nativeconnect.app/blog/mac-app-notarization-from-the-command-line/

_test_env_vars() {
    if test -z $DEVELOPER_USERNAME
    then
        echo "Error: Notarization requires you to set the DEVELOPER_USERNAME variable."
        exit -1
    fi

    if test -z $DEVELOPER_PASSWORD
    then
        echo "Error: Notarization requires you to set the DEVELOPER_PASSWORD variable."
        exit -1
    fi
}

_upload() {
    _test_env_vars
    xcrun altool \
        --notarize-app \
        --primary-bundle-id "com.pupil-labs.pupil" \
        --username "$DEVELOPER_USERNAME" \
        --password "$DEVELOPER_PASSWORD" \
        --asc-provider "R55K9ESN6B" \
        --file "$1"
}

_status() {
    _test_env_vars
    xcrun altool \
        --username "$DEVELOPER_USERNAME" \
        --password "$DEVELOPER_PASSWORD" \
        --notarization-info "$1"
}

_staple() {
    xcrun stapler staple -v $1
}

case $1 in
     upload)
        echo "Starting upload of dmg file $2"
        _upload $2
        ;;
     status)
        echo "Checking status of UUID $2"
        _status $2
        ;;
     staple)
        echo "Stapling dmg file $2"
        _staple $2
        ;;
     *)
        echo "Sorry, invalid command. Expected upload, status, or staple."
        ;;
esac