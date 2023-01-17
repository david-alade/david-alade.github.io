---
layout: post
author: David Alade
tags: [bgprouter]
permalink: /bgprouter
title: BGP Router
---

The following is a python implementation of a simple BGP (Border Gateway Protocol) router. I wrote this implementation as a part of my Networks class in the fall of 2022. The router ran against a simulator that would create neighboring routers and domain sockets to connect to them, send various messages between networks, and ask the router to “dump” its forwarding table. 


## Code

```python
import argparse
import copy
import json
import math
import select
import socket


class Router:
    relations = {}
    sockets = {}
    ports = {}
    forwarding_table = {}
    announcements = []
    revocations = []

    # This code comes from the starter code and initializes the router to set up the proper ports and sockets needed in
    # order to establish proper communication with peer networks. It also sends the handshake message.
    def __init__(self, asn, connections):
        print("Router at AS %s starting up" % asn)
        self.asn = asn
        for relationship in connections:
            port, neighbor, relation = relationship.split("-")

            self.sockets[neighbor] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets[neighbor].bind(('localhost', 0))
            self.ports[neighbor] = int(port)
            self.relations[neighbor] = relation
            self.send(neighbor,
                      json.dumps({"type": "handshake", "src": self.our_addr(neighbor), "dst": neighbor, "msg": {}}))

    # This function from the starter code is what creates the ip address for our router, which is why the last number
    # of the address is a 1.
    @staticmethod
    def our_addr(dst):
        quads = list(int(qdn) for qdn in dst.split('.'))
        quads[3] = 1
        return "%d.%d.%d.%d" % (quads[0], quads[1], quads[2], quads[3])

    # This function from the starter code sends a given message to a given ip address to allow for the proper response
    # to be received.
    def send(self, network, message):
        self.sockets[network].sendto(message.encode('utf-8'), ('localhost', self.ports[network]))

    def run(self):
        while True:
            socks = select.select(self.sockets.values(), [], [], 0.1)[0]
            for conn in socks:
                k, addr = conn.recvfrom(65535)
                srcif = None
                for sock in self.sockets:
                    if self.sockets[sock] == conn:
                        srcif = sock
                        break
                msg = k.decode('utf-8')
                print("Received message '%s' from %s" % (msg, srcif))
                self.handle_message(msg, srcif, False)
        return

    # This function develops the proper formatting needed to create an update message in order to have the proper route
    # announcements get received by our router.
    def update_message(self, msg, connection):
        aspath = copy.deepcopy(msg['msg']['ASPath'])
        aspath.insert(0, self.asn)
        message = {
            "msg":
                {
                    "netmask": msg['msg']['netmask'],
                    "ASPath": aspath,
                    "network": msg['msg']['network']
                },
            "src": self.our_addr(connection),
            "dst": connection,
            "type": 'update'
        }
        return json.dumps(message)

    # This function develops the proper formatting needed to create a withdraw message in order to have the proper route
    # announcements get withdrawn as told by a neighboring router.
    def withdraw_message(self, msg, connection):
        networks = msg['msg']
        message = {
            "msg": networks,
            "src": self.our_addr(connection),
            "dst": connection,
            "type": 'withdraw'
        }
        return json.dumps(message)

    # This function is where all the code for trying to decide if a data message can be delivered to its final spot
    # based on the entries of a data table. It handles whether a data message has a proper route that it can be
    # sent through and allows for the message of data type to go through if so.
    def forward_message(self, data_message, src):
        no_route = {
            "src": data_message['src'],
            "dst": data_message['src'],
            "type": "no route",
            "msg": {}
        }
        data_ip = data_message['dst'].split('.')
        data_ip_binary = (int(data_ip[0]) << 24) + (int(data_ip[1]) << 16) + \
                         (int(data_ip[2]) << 8) + int(data_ip[3])

        for entries in self.forwarding_table.keys():
            network = entries.split('.')
            network_binary = (int(network[0]) << 24) + (int(network[1]) << 16) + \
                             (int(network[2]) << 8) + int(network[3])
            for route in self.forwarding_table[entries]:
                subnet = route['netmask'].split('.')
                subnet_binary = (int(subnet[0]) << 24) + (int(subnet[1]) << 16) + \
                                (int(subnet[2]) << 8) + int(subnet[3])
                if data_ip_binary & subnet_binary == network_binary:
                    if self.count_prefix(network[0]) > 1:
                        longest_prefix = self.longest_prefix(data_message['dst'])
                        if longest_prefix != "tie":
                            best_network = longest_prefix
                        else:
                            best_network = entries
                    else:
                        best_network = entries
                    dest_relation = self.find_relation(data_message['dst'])
                    if dest_relation != 'no relation' or self.relations[src] == 'cust':
                        if self.relations[src] == "cust" or self.relations[dest_relation] == 'cust':
                            if len(self.forwarding_table[best_network]) > 1:
                                best_route = self.choose_dest(self.forwarding_table[best_network])
                                self.send(best_route['src'], json.dumps(data_message))
                                return
                            else:
                                best_route = self.forwarding_table[best_network][0]
                                self.send(best_route['src'], json.dumps(data_message))
                                return
        self.send(src, json.dumps(no_route))

    # This function checks which type of message is being dealt with and then takes the proper actions necessary to
    # ensure the full functionality of the message is being completed. It handles the functionality for messages of the
    # types update, data, dump, and withdraw.
    def handle_message(self, msg, src, rebuild):
        msg = json.loads(msg)
        if msg["type"] == "update":
            announcement = copy.deepcopy(msg)
            self.announcements.append(announcement)
            network_attributes = {
                "src": msg['src'],
                "netmask": msg['msg']['netmask'],
                "localpref": msg['msg']['localpref'],
                "selfOrigin": msg['msg']['selfOrigin'],
                "ASPath": msg['msg']['ASPath'],
                "origin": msg['msg']["origin"]
            }
            forwarding_table_network = self.aggregate(msg['msg']['network'], network_attributes)
            if forwarding_table_network not in self.forwarding_table:
                self.forwarding_table[forwarding_table_network] = list()
            found = False
            for neighbor in self.forwarding_table[forwarding_table_network]:
                if neighbor['src'] == network_attributes['src']:
                    found = True
            if not found:
                self.forwarding_table[forwarding_table_network].append(network_attributes)
            if not rebuild:
                if self.relations[msg['src']] == "cust":
                    for connection in self.sockets.keys():
                        if connection != msg['src']:
                            self.send(connection, self.update_message(msg, connection))

                elif self.relations[msg['src']] != "cust":
                    for connection in self.sockets.keys():
                        if self.relations[connection] == 'cust' and connection != msg['src']:
                            self.send(connection, self.update_message(msg, connection))
        elif msg["type"] == "data":
            self.forward_message(msg, src)
        elif msg['type'] == 'dump':
            table = {
                "src": self.our_addr(msg['src']),
                "dst": msg['src'],
                "type": "table",
                "msg": self.build_table()
            }
            self.send(msg['src'], json.dumps(table))
        elif msg['type'] == 'withdraw':
            self.deaggregate(msg)
            self.withdraw(msg)
            self.revocations.append(msg)
            if self.relations[msg['src']] == "cust":
                for connection in self.sockets.keys():
                    if connection != msg['src']:
                        self.send(connection, self.withdraw_message(msg, connection))
            elif self.relations[msg['src']] != "cust":
                for connection in self.sockets.keys():
                    if self.relations[connection] == 'cust' and connection != msg['src']:
                        self.send(connection, self.withdraw_message(msg, connection))

    # chooses a destination when a network has multiple neighbors that we can send to
    @staticmethod
    def choose_dest(networks):
        max_local_pref = -1
        max_local_pref_count = 0
        optimal_network = networks[0]
        for network in networks:
            if network['localpref'] > max_local_pref:
                max_local_pref = network['localpref']
                max_local_pref_count = 0
                optimal_network = network
            elif network['localpref'] == max_local_pref:
                max_local_pref_count = max_local_pref_count + 1
        if max_local_pref_count == 0:
            return optimal_network
        selforigin_count = 0
        for network in networks:
            if network['selfOrigin']:
                selforigin_count = selforigin_count + 1
                optimal_network = network
        if selforigin_count == 1:
            return optimal_network

        min_aspath = math.inf
        min_aspath_count = -1

        for network in networks:
            if len(network['ASPath']) < min_aspath:
                min_aspath = len(network['ASPath'])
                min_aspath_count = 0
                optimal_network = network
            elif len(network['ASPath']) == min_aspath:
                min_aspath_count = min_aspath_count + 10

        if min_aspath_count == 0:
            return optimal_network
        optimal_network = networks[0]
        igp_count = 0
        egp_count = 0
        unk_count = 0
        optimal = ''

        for network in networks:
            if network['origin'] == 'UNK' and (optimal != "EGP" or optimal != "IGP"):
                optimal = "UNK"
                optimal_network = network
                unk_count += 1
            elif network['origin'] == 'EGP' and optimal != "IGP":
                optimal = "EGP"
                optimal_network = network
                egp_count += 1
            elif network['origin'] == 'IGP':
                optimal = "IGP"
                optimal_network = network
                igp_count += 1
        if optimal == "IGP" and igp_count == 1:
            return optimal_network
        elif optimal == "EGP" and egp_count == 1:
            return optimal_network
        elif optimal == "UNK" and unk_count == 1:
            return optimal_network

        min_ip = math.inf
        for network in networks:
            network_ip = int(network['src'].replace('.', ''))
            if network_ip < min_ip:
                min_ip = network_ip
                optimal_network = network
        return optimal_network

    # withdraws an update from our forwarding table
    def withdraw(self, withdraw_message):
        withdraw_src = withdraw_message['src']
        dead_networks = withdraw_message['msg']

        for network in self.forwarding_table.copy():
            for neighbor in self.forwarding_table[network].copy():
                for dead_network in dead_networks:
                    if neighbor['src'] == withdraw_src and dead_network['network'] == network:
                        if len(self.forwarding_table[network]) == 1:
                            del self.forwarding_table[network]
                            self.remove_announcement(network, neighbor['src'])
                        else:
                            self.remove_announcement(network, neighbor['src'])
                            neighbor.remove(neighbor)

    # removes an announcement from our list of announcements that contains our list of updates
    def remove_announcement(self, network, peer):
        for announcement in self.announcements:
            if announcement['src'] == peer and announcement['msg']['network'] == network:
                self.announcements.remove(announcement)
                break

    # finds a relation or lack thereof for a given network through our list of neighbors
    def find_relation(self, network):
        network_pre = network.split(".")
        for relation in self.relations.keys():
            relation_pre = relation.split('.')
            if network_pre[0] == relation_pre[0]:
                return relation
        return "no relation"

    # finds the longest prefix match of a given destination from our forwarding table
    def longest_prefix(self, dst):
        longest_prefix = 0
        net_pre = dst.split(".")[0]
        dst_binary = '.'.join([bin(int(x) + 256)[3:] for x in dst.split('.')])
        best_network = ""
        for entry in self.forwarding_table.keys():
            net_list = entry.split('.')
            if net_list[0] == net_pre:
                network_binary = '.'.join([bin(int(x) + 256)[3:] for x in entry.split('.')])
                match_length = self.count_similar(dst_binary, network_binary)
                if match_length > longest_prefix:
                    longest_prefix = match_length
                    best_network = entry

        return best_network

    # counts the number the provided prefix in the forwarding table
    def count_prefix(self, prefix):
        count = 0
        for network in self.forwarding_table.keys():
            network_prefix = network.split(".")
            network_prefix = network_prefix[0]
            if network_prefix == prefix:
                count += 1
        return count

    # counts the length of the prefix match for a given network and destination
    @staticmethod
    def count_similar(dest_binary, network_binary):
        count = 0
        for i in range(len(dest_binary)):
            if dest_binary[i] == network_binary[i]:
                count += 1
            else:
                return count

    # looks for a network in our routing table that could possibly be aggregated into the given network.
    # If not, returns the given network and doesn't mutate the subnet mask.
    def aggregate(self, network, attributes):
        network_quad = network.split('.')
        network_quad = network_quad[2]
        for entry in self.forwarding_table.keys():
            for neighbor in self.forwarding_table[entry]:
                key_quad = entry.split('.')
                key_quad = key_quad[2]
                mask_quad = neighbor['netmask'].split('.')
                hop = self.check_hop(mask_quad, key_quad, network_quad)
                if ((abs(int(network_quad) - int(key_quad)) == 1) or hop) and self.check_attributes(attributes,
                                                                                                    neighbor) and (
                        attributes['src'] == neighbor['src']):
                    entry_int = int(entry.replace('.', ""))
                    network_int = int(network.replace('.', ""))

                    edit_mask = neighbor['netmask'].split('.')

                    for i in range(4):
                        if edit_mask[i] == "0":
                            edit_mask[i - 1] = str(int(edit_mask[i - 1]) - 1)
                            break
                        if i == 3:
                            edit_mask[i] = str(int(edit_mask[i]) - 1)

                    neighbor['netmask'] = '.'.join(edit_mask)
                    if entry_int < network_int:
                        return entry
                    else:
                        del self.forwarding_table[entry]
                        return network
        return network

    # performs a check on the given network attributes to check if they have the same fields
    @staticmethod
    def check_attributes(attributes1, attributes2):
        return (attributes1['localpref'] == attributes2['localpref']) and \
               attributes1['selfOrigin'] == attributes2['selfOrigin'] and \
               (attributes1['ASPath'] == attributes2['ASPath']) and (attributes1['origin'] == attributes2['origin'])

    # checks if a given network is within one hop of the networks that we have available to us.
    @staticmethod
    def check_hop(mask_quad, key_quad, net_quad):
        return ((int(key_quad) + abs(int(mask_quad[2]) - 255)) + 1) == int(net_quad)

    # builds the dump table to be sent to neighbors from our forwarding table.
    def build_table(self):
        table_msg = []
        for entry in self.forwarding_table.keys():
            for neighbor in self.forwarding_table[entry]:
                output = {
                    'origin': neighbor['origin'],
                    'localpref': neighbor['localpref'],
                    'network': entry,
                    'ASPath': neighbor['ASPath'],
                    'netmask': neighbor['netmask'],
                    'peer': neighbor['src'],
                    'selfOrigin': neighbor['selfOrigin']
                }
                table_msg.append(output)
        return table_msg

    # deaggregates our table when withdraw is called so that the forwarding table can be recalibrated with the withdrawn
    # message no included.
    def deaggregate(self, withdraw_msg):
        copy_announce = copy.deepcopy(self.announcements)
        for announcement in copy_announce:
            for network in withdraw_msg['msg']:
                if network['network'] == announcement['msg']['network'] and withdraw_msg['src'] == announcement['src']:
                    self.announcements.remove(announcement)
        loop_copy_announce = copy.deepcopy(self.announcements)
        self.forwarding_table = {}
        self.announcements = []

        for announcement in loop_copy_announce:
            self.handle_message(json.dumps(announcement), "", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='route packets')
    parser.add_argument('asn', type=int, help="AS number of this router")
    parser.add_argument('connections', metavar='connections', type=str, nargs='+', help="connections")
    args = parser.parse_args()
    router = Router(args.asn, args.connections)
    router.run()

```