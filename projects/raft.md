---
layout: post
author: David Alade
tags: [raft, python]
permalink: /raft
title: Raft
---

The following is a python implementation of the Raft consensus protocol for a distributed key-value store. I wrote this implementation as apart of my Networks class in the fall of 2022. The implentation represents the protocol for a replica in the cluster to follow to keep the whole cluster updated as a comprehensize database. The code was tested through a simulator that would send various put, and get requests to replicas while also randomly killing replicas, and leaders. 

## Code

```python
#!/usr/bin/env python3

import argparse, socket, time, json, select, struct, sys, math, os, random
import copy

BROADCAST = "FFFF"


class Replica:
    def __init__(self, port, id, others):
        self.port = port
        self.id = id
        self.others = others
        self.state = "follower"
        self.leader_id = BROADCAST

        self.state_machine = {}
        self.current_term = 0
        self.voted = False
        self.log = []
        self.log_confirmations = {}
        self.last_heartbeat = 0
        self.election_timeout = random.uniform(.150, .300)
        self.election_start = 0

        self.commit_index = 0
        self.last_applied = 0
        self.votes = 0
        self.worklist = []
        self.sent = {}

        # leader only
        self.prev_log_index = -1
        self.prev_log_term = 0
        self.next_index = {}
        self.match_index = {}
        self.store_time = {}
        self.timeout = False

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('localhost', 0))
        print("Replica %s starting up" % self.id, flush=True)
        hello = {"src": self.id, "dst": BROADCAST, "leader": BROADCAST, "type": "hello"}
        self.send(hello)
        print("Sent hello message: %s" % hello, flush=True)

    def send(self, message):
        print("Sending message: %s" % message, flush=True)
        self.socket.sendto(json.dumps(message).encode('utf-8'), ('localhost', self.port))

    def run(self):
        while True:
            socks = select.select([self.socket], [], [], 0.1)[0]

            if self.state == 'leader':
                self.check_follower()

            if self.socket in socks:
                data, addr = self.socket.recvfrom(65535)
                msg = data.decode('utf-8')
                print("Received message '%s'" % msg, flush=True)
                msg_dict = json.loads(msg)

                self.handle_message(msg_dict)

            if self.state == 'follower' and time.time() - self.last_heartbeat > .5 and not self.voted:
                self.state = 'candidate'
                self.start_election()

            if self.state == 'candidate':
                if time.time() - self.election_start > self.election_timeout:
                    self.start_election()

            if self.state == 'leader' and time.time() - self.last_heartbeat >= .1:
                self.heartbeat()

    def start_election(self):
        print("STARTING VOTE")
        self.current_term += 1
        self.votes = 1
        self.request_vote()
        self.election_start = time.time()
        self.election_timeout = random.uniform(.15, .4)

    def handle_message(self, msg):
        print("HANDLING" + str(msg), flush=True)

        if self.leader_id != BROADCAST:
            while len(self.worklist) != 0:
                handle = self.worklist.pop(0)
                # print("HANDLING FROM WORKLIST " + str(handle), flush=True)
                self.handle_message(handle)

        if msg['type'] == 'get':
            self.handle_get(msg)
        elif msg['type'] == 'put':
            self.handle_put(msg)

        elif msg['type'] == 'request_vote':
            self.handle_request_vote(msg)

        elif msg['type'] == 'append_entry':
            self.handle_append_entry(msg)

        elif msg['type'] == 'vote':
            if self.state == 'candidate':
                self.votes += 1
                majority = int(len(self.others) / 2) + 1
                print(str(self.votes) + " votes so far.", flush=True)

                if self.votes >= majority:
                    print('I AM THE LEADER ' + self.id, flush=True)
                    self.state = 'leader'
                    self.leader_id = self.id
                    self.heartbeat()
                    self.votes = 0
                    if len(self.log) == 0:
                        for follower in self.others:
                            self.next_index[follower] = 0
                    else:
                        for follower in self.others:
                            self.next_index[follower] = self.log[-1]['index'] + 1
            else:
                return

        elif msg['type'] == 'stored':
            if self.state == 'leader':
                self.store_time[msg['src']] = time.time()
                if self.log_confirmations[msg['index']] != -1:
                    self.log_confirmations[msg['index']] += 1
                    if msg['index'] > self.match_index.get(msg['src'], -1):
                        self.match_index[msg['src']] = msg['index']
                    majority = int(len(self.others) / 2) + 1
                    if self.log_confirmations[msg['index']] >= majority:
                        self.update_log(msg['index'])
                        self.log_confirmations[msg['index']] = -1

        elif msg['type'] == 'rejected':
            self.next_index[msg['src']] -= 1
            index = self.next_index[msg['src']]
            if index < 0:
                entry = self.log[0]
            else:
                entry = self.log[index]

            message = {
                'src': self.id,
                'dst': msg['src'],
                'type': "new_log",
                'leader': self.leader_id,
                'log': self.log,
                'highest_log': self.commit_index,
                'term': self.current_term
            }
            self.send(message)
        elif msg['type'] == 'new_log':
            self.log = msg['log']

    def handle_get(self, msg):
        value = self.state_machine.get(msg['key'], '')
        message = {
            "src": self.id,
            "dst": msg['src'],
            "leader": self.leader_id,
            "MID": msg['MID'],
            'term': self.current_term
        }

        if self.state != 'leader':
            message['type'] = 'redirect'
            self.send(message)

        if self.state == 'follower' or self.state == 'leader':
            message['type'] = 'ok'
            message['value'] = value
            if value is not None:
                self.send(message)
        else:
            # print("ADDING TO WORK LIST IN GET" + str(msg))
            self.worklist.append(msg)

    def heartbeat(self):
        self.last_heartbeat = time.time()
        for follower in self.others:
            message = {
                "src": self.id,
                "dst": follower,
                'type': 'append_entry',
                'leader': self.id,
                'log': [],
                'term': self.current_term,
                'highest_log': len(self.state_machine)
            }
            self.send(message)

    def handle_request_vote(self, msg):
        if len(self.log) > 0:
            latest_log = self.log[-1]

            if msg['lastLogTerm'] != latest_log['term']:
                up_to_date = latest_log['term'] > msg['lastLogTerm']
            else:
                up_to_date = latest_log['index'] > msg['lastLogIndex']
        else:
            up_to_date = False

        if self.state == 'follower' and not self.voted and not up_to_date:
            message = {
                'src': self.id,
                'dst': msg['src'],
                'type': 'vote',
                'leader': BROADCAST,
                'term': self.current_term
            }
            self.voted = True
            self.current_term += 1
            self.send(message)
        else:
            return

    def request_vote(self):
        for peer in self.others:
            if len(self.log) > 0:
                lastLogIndex = self.log[-1]['index']
                lastLogTerm = self.log[-1]['term']
            else:
                lastLogIndex = 0
                lastLogTerm = self.current_term
            message = {
                'src': self.id,
                'dst': peer,
                'type': 'request_vote',
                'leader': BROADCAST,
                'term': self.current_term,
                'lastLogIndex': lastLogIndex,
                'lastLogTerm': lastLogTerm
            }
            self.send(message)

    def handle_append_entry(self, msg):
        print("IN APPEND ENTRY")
        if self.state == 'follower':
            self.election_timeout = random.uniform(.15, .3)
            self.last_heartbeat = time.time()

            if len(msg['log']) == 0:  # a heartbeat
                print("append 1", flush=True)
                self.votes = 0
                self.leader_id = msg['src']
                self.state = 'follower'
                self.voted = False

            else:  # an actual entry
                message = {
                    'src': self.id,
                    'dst': msg['src'],
                    'type': 'stored',
                    'index': msg['log']['index'],
                    'leader': self.leader_id
                }

                if len(self.log) > 0:
                    for ent in reversed(self.log):
                        if ent['term'] == msg['log']['prevLogTerm'] and ent['index'] == msg['log']['prevLogIndex']:
                            self.log.append(msg['log'])
                            self.commit_index = msg['highest_log']
                            self.update_log(self.commit_index)
                            self.send(message)
                            return

                    print(self.log)
                    reject_message = {
                        'src': self.id,
                        'dst': self.leader_id,
                        'leader': self.leader_id,
                        'type': 'rejected',
                        'term': self.current_term
                    }

                    self.send(reject_message)
                else:
                    self.log.append(msg['log'])
                    self.commit_index = msg['highest_log']
                    self.update_log(self.commit_index)
                    self.send(message)
        elif self.state == 'candidate' or self.state == 'leader':
            if msg['term'] >= self.current_term:
                self.state = 'follower'
                if len(msg['log']) != 0:
                    self.handle_append_entry(msg)

    def handle_put(self, msg):
        message = {
            "src": msg['dst'],
            "dst": msg['src'],
            "leader": self.leader_id,
            "MID": msg['MID']
        }

        if self.state == 'follower':
            if self.leader_id != BROADCAST:
                message['type'] = 'redirect'
                self.send(message)
            else:
                # print("ADDING TO WORKLIST IN PUT" + str(msg), flush=True)
                self.worklist.append(msg)

        elif self.state == 'leader':
            log_entry = {
                'term': self.current_term,
                'key': msg['key'],
                'value': msg['value'],
                'type': 'ok',
                'index': len(self.log)
            }

            new_entry = {**log_entry, **message, 'prevLogIndex': self.prev_log_index,
                         'prevLogTerm': self.current_term}

            self.log.append(new_entry)
            self.prev_log_index += 1
            self.log_confirmations[log_entry['index']] = 0
            self.append_entry(new_entry)

        elif self.state == 'candidate':
            # print("ADDING TO WORKLIST IN PUT AS CANDIDATE" + str(msg), flush=True)
            self.state = 'follower'
            self.worklist.append(msg)

    def append_entry(self, entry):
        for follower in self.others:
            message = {
                'src': self.id,
                'dst': follower,
                'type': "append_entry",
                'leader': self.leader_id,
                'log': entry,
                'highest_log': self.commit_index,
                'term': self.current_term
            }
            self.send(message)

    def update_log(self, index):
        if self.state == 'leader':
            log = self.log[index]
            self.state_machine[log['key']] = log['value']
            print("KEY " + log['key'] + " ADDED AS VALUE: " + log['value'], flush=True)

            message = {
                'src': self.id,
                'dst': log['dst'],
                'leader': self.leader_id,
                'type': 'ok',
                'MID': log['MID']
            }
            self.commit_index = index
            self.send(message)
        if self.state == 'follower':
            while self.last_applied < self.commit_index and self.last_applied < len(self.log) and len(self.log) > 0:
                print("CATCHING UP", flush=True)
                commit = self.log[self.last_applied]
                self.state_machine[commit['key']] = commit['value']
                self.last_applied += 1

            print("DONE CATCHING UP", flush=True)

    def check_follower(self):
        for follower in self.others:
            response_time = self.store_time.get(follower)
            if response_time is not None:
                if time.time() - response_time > 1.5:
                    print("REMOVING " + follower + "BECAUSE WE HAVEN'T HEARD FROM IT")
                    self.others.remove(follower)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run a key-value store')
    parser.add_argument('port', type=int, help="Port number to communicate")
    parser.add_argument('id', type=str, help="ID of this replica")
    parser.add_argument('others', metavar='others', type=str, nargs='+', help="IDs of other replicas")
    args = parser.parse_args()
    replica = Replica(args.port, args.id, args.others)
    replica.run()

```