# Pupil Time Sync Protocol

Protocol version: v1
Protocol status: draft

The Pupil Time Sync -- hereinafter referred to as _PTS_ -- protocol consists of two parts:
1. Clock service discovery
2. Time synchronization

There are two types of actors -- hereinafter referred to as _PTS actors_ -- who
can participate:
1. The _clock service_
2. The _clock follower_

Each clock service has a _rank_ (see below for details). The clock service with
the highest rank becomes _clock master_. The clock followers synchronize their
time according to the clock master.

## Clock service discovery

This part has been designed such that each PTS actor can be a clock service,
a clock follower, or both. PTS actors find each other using the [ZeroMQ Realtime
Exchange Protocol](https://rfc.zeromq.org/spec:36/ZRE). We recommend the usage
of existing libraries (e.g. [zyre](https://github.com/zeromq/zyre), [Pyre](https://github.com/zeromq/pyre)) that implement the ZRE protocol.

### Synchronization scope

All PTS actors that should be synchronized SHALL join a ZRE group -- hereinafter
referred to as _PTS group_. The name of the PTS group is composed of a user-definable
prefix -- by default `default` -- and the fixed string `-time_sync-v1`. Therefore the
default PTS group name is `default-time_sync-v1`.

All clock services SHALL SHOUT their announcements (see below) into the PTS group.

### Clock service anouncements

A clock service anouncement is a two-frame ZRE message that is SHOUTed into the
PTS group:
1. The first frame contains the Python string representation of the service's rank
2. The second frame contains the Python string representation of service's network port

Messages that are not formatted in this way SHOULD be ignored.

A clock service anouncement SHALL happen in the following cases:
* The clock service's own rank changes
* There is a JOIN event in the PTS group
* The clock service's network port changes

It is allowed to send only one announcement if the events happen in a short
series of time -- e.g. the client will receive all JOIN events for the current
network state on entering the ZRE network.

### Clock service rank

The clock service rank is a positive float that is linear combination of the
following parts:

1. `base_bias`: A user-definable positive float with default value `1.0`
2. `has_been_master`: `1.0` if the clock service has been clock master before
    else `0.0`
3. `has_been_synced`: `1.0` if the service's clock has been synced to an other
    clock master before else `0.0`
4. `tie_breaker`: A random float between `0.0` and `1.0` with enough bits to be
    unique and act as tie breaker.^1

The total rank SHOULD be calculated using this formular:
```
4*base_bias + 2*has_been_master + has_been_synced + tie_breaker
```

^1: This needs a better definition. The reference implementation uses Python's
    [`random.random()`](https://docs.python.org/3/library/random.html#random.random)
    function to generate the `tie_breaker`.

### Clock service invalidation

A clock service is invalidated by leaving the PTS group or exiting the ZRE network.

### Clock master selection

The clock service with the highest rank becomes _clock master_ of its PTS group.
All clock followers must evaluate the clock master on following occasions:
* On receiving a new clock service announcement
* On a clock service leaving the PTS group or exiting the ZRE network

Evaluating the clock master means to compare all clock services' ranks and synchronize
its own clock with the clock master (see beow for details).


## Time synchronization

The time synchronization is based on the [Network Time Protocol](https://en.wikipedia.org/wiki/Network_Time_Protocol). The clock follower sends a series of small messages
to the clock master. The clock master answers with its own time. Upon receiving
the clock master's answer the follower knows the round trip time to the service
and is able to change its clock appropriately.

### Timestamp unit

Timestamps are 64-bit little-endian floats in seconds.

### Clock service

The clock service is a simple TCP server that sends its own current timestamp
upon receiving the message `sync` from a follower. The TCP server's network port
SHALL be announced as part of the clock serviceannouncement (see above).

### Clock follower

The clock follower calculates its clock's offset and offset-jitter regularly in
the following manner:

- Open a TCP connection to the time service.
- Repeat the following steps 60 times:
    - Measure the follower's current timestamp `t0`
    - Send `sync` to the clock master
    - Receive the clock master's response as `t1` and convert it into a float
    - Measure the follower's current timestamp as `t2`
    - Store entry `t0`, `t1`, `t2`
- Sort entries by _roundtrip time_ (`t2 - t0`) in ascending order
- Remove last 30% entries, i.e. remove outliers
- Calculate _offset_ for each entry: `t0 - (t1 + (t2 - t0) / 2)`
- Calculate _mean offset_
- Calculate _offset variance_
- Use mean offset as _offset_ and clock variance as _offset jitter_
- Adjust the follower's clock according to the offset and the offset jitter

