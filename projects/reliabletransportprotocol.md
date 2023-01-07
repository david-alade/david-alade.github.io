---
layout: post
author: David Alade
tags: [reliabletransportprotocol]
permalink: /reliabletransportprotocol
title: Reliable Transport Protocol
---

 Flood it is a simple game that I wrote in in the spring of 2022 for my Fundamentals of Computer Science 2 class. It uses the java image library and it runs using the Khoury tester library function "big-bang".

When a player clicks a cell of a certain color, all cells of that color that are connected to the "flooded" componenet at the top left of the board become flooded. The goal of the game is to flood the entire board with a single color within the given amount of moves. 

<details>
<summary>Send client</summary>

{% highlight python %}

import argparse, socket, time, json, select, struct, sys, math, hashlib
import ast
from _socket import timeout

DATA_SIZE = 1375


class Sender:
    done = False
    packets = []
    actual = []
    acks = []

    # This function initializes the values we use for our Sender.
    def __init__(self, host, port):
        self.host = host
        self.remote_port = int(port)
        self.log("Sender starting up using port %s" % self.remote_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', 0))
        self.waiting = False

    # This function takes in a message and displays it in the simulator.
    def log(self, message):
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    # This function takes in a message and sends it to the proper location.
    def send(self, message):
        self.socket.sendto(json.dumps(message).encode('utf-8'), (self.host, self.remote_port))

    # This function runs our Sender.
    def run(self):
        while True:
            # This if-statement checks to ensure that we have packets left and if our sender is done.
            if len(self.packets) > 0 and self.done:
                self.retransmit()
            sockets = [self.socket, sys.stdin] if not self.waiting else [self.socket]
            socks = select.select(sockets, [], [], 0.1)[0]
            for conn in socks:
                if conn == self.socket:

                    # This will loop through the packets that are being sent in the program.
                    for x in range(len(self.packets)):
                        try:
                            conn.settimeout(.1)
                            k, addr = conn.recvfrom(65535)

                            # This try-except checks the hash of our message and handles the error when our JSON gets
                            # corrupted accordingly.
                            try:
                                msg = json.loads(k.decode('utf-8'))
                            except json.decoder.JSONDecodeError:
                                continue
                            if 'number' not in msg:
                                sys.exit(0)
                            if "hash" not in msg or 'type' not in msg:
                                continue
                            for packet in self.packets:
                                if packet[2] == msg['hash'] and packet[0][12:19] == msg['number']:
                                    self.packets.remove(packet)
                        except timeout:
                            self.retransmit()
                    self.waiting = False

                # This else-if handles the case when we have data that we still need to be gathering from the simulator.
                elif conn == sys.stdin:
                    while len(self.packets) < 4 and not self.done:
                        data = sys.stdin.read(DATA_SIZE)
                        self.send_message(data)
                    self.waiting = True
        return

    # This function takes in the data from a packet, sends it to the receiver and adds the sent packet to our list of
    # packets.
    def send_message(self, data):
        msg_hash = hashlib.sha256(data.encode()).hexdigest()

        if len(data) == 0:
            msg = {"type": "msg", "data": "finished"}
            self.send(msg)
            self.packets.append(("finished", time.time(), msg_hash))
            self.done = True
            return
        msg = {"type": "msg", "data": data, 'hash': msg_hash}
        self.send(msg)
        self.packets.append((data, time.time(), msg_hash))

    # This function checks to see if our data has been transmitted within a certain time frame and handles the
    # retransmission of the packet accordingly.
    def retransmit(self):
        packet = self.packets[0]
        if time.time() - packet[1] > 1.0:
            # msg = {"type": "msg", "data": packet[0]}
            # self.log("Retransmitting message '%s'" % msg)
            self.packets.remove(packet)
            self.send_message(packet[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='send data')
    parser.add_argument('host', type=str, help="Remote host to connect to")
    parser.add_argument('port', type=int, help="UDP port number to connect to")
    args = parser.parse_args()
    sender = Sender(args.host, args.port)
    sender.run()
{% endhighlight %}

</details>



<details>
<summary>Recieve client</summary>

{% highlight python %}

import argparse, socket, time, json, select, struct, sys, math, hashlib

class Receiver:
    sequence_numbers = {}
    work_list = []
    next_up = 0
    finished = False

    # This function initializes the values we use for our Receiver.
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', 0))
        self.port = self.socket.getsockname()[1]
        self.log("Bound to port %d" % self.port)

        self.remote_host = None
        self.remote_port = None

    # This function takes in a message and sends it to the proper location.
    def send(self, message):
        self.socket.sendto(json.dumps(message).encode('utf-8'), (self.remote_host, self.remote_port))

    # This function takes in a message and displays it in the simulator.
    def log(self, message):
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    # This function runs our Receiver.
    def run(self):
        while True:
            socks = select.select([self.socket], [], [])[0]
            for conn in socks:
                data, addr = conn.recvfrom(65535)

                # Grab the remote host/port if we don't already have it
                if self.remote_host is None:
                    self.remote_host = addr[0]
                    self.remote_port = addr[1]

                msg = json.loads(data.decode('utf-8'))
                # self.log("Received message %s" % msg)

                # This initial if-statement checks to see if the program has finished sending data from the sender.
                if msg['data'] != "finished":
                    number = msg["data"][12:19]
                    hash_msg = hashlib.sha256(msg['data'].encode()).hexdigest()

                    # This if-statement is where we check if the message we receive has the correct hash value assigned
                    # it so that we can determine if a packet has become corrupted or not.
                    if hash_msg == msg['hash']:
                        self.send({"type": "ack", "number": number, 'hash': hash_msg})

                        # In this if-statement we check to see if a sequence number has already been delievered and if
                        # it hasn't, we add it to a list of the other sequence numbers.
                        if number not in self.sequence_numbers:
                            self.sequence_numbers[number] = msg["data"]
                            self.work_list.append(int(number))
                            self.work_list.sort()

                            if len(self.work_list) > 0:
                                while self.work_list[0] == self.next_up:
                                    self.next_up += 1
                                    self.work_through()
                                    if len(self.work_list) == 0:
                                        break
                        else:
                            pass
                            # self.log("Received data duplicate message %s" % msg)
                    else:
                        pass
                        # self.log("Received corrupted message %s" % msg)

                else:
                    if len(self.work_list) == 0:
                        self.send({"type": "ack"})
                    elif self.work_list[0] == self.next_up:
                        key = ("%07d" % self.work_list[0])
                        self.work_list.pop(0)
                        print(self.sequence_numbers[key], end='', flush=True)

        return

    # This function goes through our work list of sequence numbers and takes them out from the list starting from the
    # first element.
    def work_through(self):
        key = ("%07d" % self.work_list[0])
        self.work_list.pop(0)
        print(self.sequence_numbers[key], end='', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='receive data')
    args = parser.parse_args()
    sender = Receiver()
    sender.run()

{% endhighlight %}

</details>