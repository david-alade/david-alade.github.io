---
layout: post
author: David Alade
tags: [ftpclient, python]
permalink: /ftpclient
title: FTP Client
---
The following is a python implementation of the File Transport Protocol. I wrote this implementation as a part of my Networks class in the fall of 2022. This client connected to a northeastern FTP server and had various FTP commands run on it with various files to test its functionality. 

## Code

```python
#!/usr/bin/env python3

import os
import socket
from urllib.parse import urlparse
import argparse

# Commands to be sent through the control channel
mode = "MODE S"
stru = "STRU F"
retr = "RETR "
type_command = "TYPE I"
stor = "STOR "
quit_command = "QUIT"
pasv = "PASV"

# Maps a command line command to its control channel command counterpart
commands = {
    'ls': "LIST ",
    'mkdir': "MKD ",
    'rm': "DELE ",
    'rmdir': "RMD ",
}


# handles arg parsing and passes the raw url, path and operation to the client program functions
def main():
    parser = argparse.ArgumentParser("Initiates command line preferences")
    parser.add_argument("operation", type=str)
    parser.add_argument("params",
                        help="Parameters for the given operation. Will be one or two paths and/or URLs.",
                        type=str, nargs='+')
    args = parser.parse_args()
    operation = args.operation
    params = args.params
    param1 = params[0]

    if len(params) > 1:
        param2 = params[1]
        client_program(operation, param1, param2)
    else:
        client_program_1(operation, param1)


# Handles all the communication with the ftp server when two parameters are passed in
def client_program(operation, param1, param2):
    url = urlparse(param2)
    path = param1
    marker = "path->url"
    port = 21

    if url.port:
        port = url.port

    if ":" in param1:
        url = urlparse(param1)
        path = param2
        marker = "url->path"

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((url.hostname, port))
    client_socket.recv(1024).decode()
    welcome_message(url, client_socket)

    if operation == 'cp':
        copy(marker, client_socket, url, path, False)
    if operation == 'mv':
        copy(marker, client_socket, url, path, True)


# Handles all the communication with the ftp server when one parameter is passed in
def client_program_1(operation, param1):
    url = urlparse(param1)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 21

    if url.port:
        port = url.port

    client_socket.connect((url.hostname, port))
    print(client_socket.recv(1024).decode())

    welcome_message(url, client_socket)

    if operation == "ls" or operation == "rm":
        data = send_message(pasv, client_socket)
        numbers = strip_ip(data)
        ip = ".".join(numbers[:4])
        data_port = (int(numbers[4]) << 8) | int(numbers[5])
        print(data_port)
        client_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_data_socket.connect((ip, data_port))
        send_message_path(commands[operation], url, client_socket)
        if operation == 'ls':
            data_channel = client_data_socket.recv(2048).decode()
            print(data_channel)

        send_message(quit_command, client_socket)
    else:
        send_message_path(commands[operation], url, client_socket)


# sends a message through the supplied socket and returns the data that the server responded with
def send_message(message, c_socket):
    message = message + "\r\n"
    c_socket.send(message.encode())
    data = c_socket.recv(1024).decode()
    print(data)
    return data


# helper to send messages that use the url path
def send_message_path(command, url, c_socket):
    send_message(command + url.path, c_socket)


# strips the ip string of everything and turns it into a list of numbers as strings
def strip_ip(string):
    string = string[string.index("(") + 1:]
    string = string.replace(").\r\n", "")
    numbers = string.split(",")
    return numbers


# sends the welcome message through the socket that sends in the username and password to connect to the ftp server
def welcome_message(url, c_socket):
    if url.username:
        user = "USER " + url.username
        send_message(user, c_socket)

    if url.password:
        password = "PASS " + url.password
        send_message(password, c_socket)


# handles both the moving and copying of files to and from the url. If the "move" parameter is true then the file being
# copied will be removed
def copy(marker, c_socket, url, path, move):
    data = send_message(pasv, c_socket)
    numbers = strip_ip(data)
    ip = ".".join(numbers[:4])
    data_port = (int(numbers[4]) << 8) | int(numbers[5])
    client_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_data_socket.connect((ip, data_port))

    if marker == "url->path":
        send_message(type_command, c_socket)
        send_message(mode, c_socket)
        send_message(stru, c_socket)
        send_message_path(retr, url, c_socket)

        with open(path, "wb") as file:
            download = client_data_socket.recv(8224)
            if move:
                send_message_path(commands["rm"], url, c_socket)
            file.write(download)
            client_data_socket.close()
            send_message(quit_command, c_socket)

    if marker == "path->url":
        send_message(type_command, c_socket)
        send_message(mode, c_socket)
        send_message(stru, c_socket)
        send_message(stor + url.path, c_socket)

        with open(path, "rb") as file:
            send = file.read()
            client_data_socket.send(send)
            if move:
                os.remove(path)
            client_data_socket.close()

        send_message(quit_command, c_socket)


if __name__ == '__main__':
    main()


```