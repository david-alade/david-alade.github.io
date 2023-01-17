---
layout: post
author: David Alade
tags: [wordguesser, python]
permalink: /wordguesser
title: Word Guesser
---

The following is a python implementation of a simple word guesser. I wrote this implementation as a part of my Networks class in the fall of 2022. This program simply sends guesses to a server through a socket, and the server responds with information of which letters are in the right spot for the guessed word. This information is used to narrow down the word to guess.
## Code

```python
#!/usr/bin/env python3

import socket
import argparse
import sys
import json
import ssl


# Main class that handles the arg parsing and calls the client_program function with the given args
def main():
    if len(sys.argv) < 2:
        print("Please input the hostname and northeaster username")

    parser = argparse.ArgumentParser('Initiates command lind preferences')
    parser.add_argument('-p', '--port', type=int, default=27993)
    parser.add_argument('-s', action='store_true')
    parser.add_argument("hostname", type=str)
    parser.add_argument("northeasternUsername", type=str)
    args = parser.parse_args()

    hostname = args.hostname
    northeastern = args.northeasternUsername
    port = args.port
    s_flag = args.s
    # if it is ssl then the program should use this port
    if s_flag:
        port = 27994

    client_program(port, s_flag, hostname, northeastern)


# client program that creates the socket and handles the sending and receiving of messages
def client_program(port, tls, hostname, northeastern):
    context = ssl.create_default_context()
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # handles the case where the socket has to be tls encrypted
    if tls:
        client_socket = context.wrap_socket(client_socket, server_hostname=hostname)

    client_socket.connect((hostname, port))
    client_socket.send((json.dumps(send_message("hello", northeastern)) + "\n").encode(encoding="ascii"))
    data = client_socket.recv(1024).decode(encoding="ascii")
    data_dict = json.loads(data)
    perm_id = data_dict['id']
    bad_letters = []
    good_letters = []
    potential_words = []
    word_list = open('project1-words.txt', 'r').readlines()

    # list of words that use every letter in the alphabet
    all_letters = ['fjord', 'gucks', 'nymph', 'vibex', 'waltz', 'nuque']

    # Send guesses to the server using the all_letters list. This is to figure out which letters are in the word to
    # narrow it down

    for word in all_letters:
        client_socket.send((json.dumps(send_guess(perm_id, word)) + "\n").encode(encoding="ascii"))
        data = client_socket.recv(1024).decode(encoding="ascii")
        data_dict = json.loads(data)

        for x in range(5):
            if data_dict['guesses'][-1]['marks'][x] == 0:
                if data_dict['guesses'][-1]['word'][x] not in bad_letters:
                    bad_letters.append(data_dict['guesses'][-1]['word'][x])
            if data_dict['guesses'][-1]['marks'][x] == 1:
                if data_dict['guesses'][-1]['word'][x] not in good_letters:
                    good_letters.append(data_dict['guesses'][-1]['word'][x])
            if data_dict['guesses'][-1]['marks'][x] == 2:
                if data_dict['guesses'][-1]['word'][x] not in good_letters:
                    good_letters.append(data_dict['guesses'][-1]['word'][x])

    # Makes sure that all valid letters are not in the bad_letters list
    for letter in good_letters:
        if letter in bad_letters:
            bad_letters.remove(letter)

    # Only adds words that don't have a bad letter to the list of potential words
    for word in word_list:
        if not any(ele in word for ele in bad_letters):
            potential_words.append(word)

    # Send guesses of potential words to the server to get the flag
    for word in potential_words:
        client_socket.send((json.dumps(send_guess(perm_id, word.strip())) + "\n").encode(encoding="ascii"))
        data = client_socket.recv(4096).decode(encoding="ascii")
        data_dict = json.loads(data)

        if data_dict['type'] == 'bye':
            client_socket.close()
            print(data_dict['flag'])
            break


# helper method to send the hello message as a dictionary
def send_message(messageType, username):
    message = {
        "type": messageType,
        "northeastern_username": username
    }

    return message


# helper method to send guesses to the server
def send_guess(id, word):
    message = {
        "type": "guess",
        "id": id,
        "word": word
    }

    return message


if __name__ == '__main__':
    main()
```
