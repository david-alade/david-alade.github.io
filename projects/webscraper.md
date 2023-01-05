---
layout: post
author: David Alade
tags: [webscraper, python]
permalink: /webscraper
title: Web Scraper
---

 Flood it is a simple game that I wrote in in the spring of 2022 for my Fundamentals of Computer Science 2 class. It uses the java image library and it runs using the Khoury tester library function "big-bang".

When a player clicks a cell of a certain color, all cells of that color that are connected to the "flooded" componenet at the top left of the board become flooded. The goal of the game is to flood the entire board with a single color within the given amount of moves. 

## Code

```python
#!/usr/bin/env python3

import argparse
import copy
import socket
import ssl
from html.parser import HTMLParser
from _socket import timeout

DEFAULT_SERVER = "proj5.3700.network"
DEFAULT_PORT = 443
frontier = []
visited = []
first_run = []
flags = []
token = ''
logged_in = False
crawling = False
sessionid = None
cookie = None


# This is the class where we set up the variables and methods needed in order for our crawler to function as intended.
class Crawler:
    def __init__(self, args):
        self.server = args.server
        self.port = args.port
        self.username = args.username
        self.password = args.password
        self.session_cookie = None
        self.all_flags = False

    # This method sends our initial message, logs into the fakebook, and then begins ti crawl through the site in order
    # to find all the flags.
    def run(self):
        self.send_get_message("/", "")
        home_page_dict = self.login()
        self.crawl(home_page_dict)

    # This method sends the GET request to the server, adds whichever link it used for the request to our array of
    # visited links, and then returns the data that we receive.
    def send_get_message(self, path, cookie):
        request = "GET " + path + " HTTP/1.1\n" + "Host: " + DEFAULT_SERVER + "\n" + cookie + '\r\n\r\n'
        context = ssl.create_default_context()
        mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mysocket = context.wrap_socket(mysocket, server_hostname=self.server)
        mysocket.connect((self.server, self.port))
        mysocket.send(request.encode('ascii'))
        visited.append(path)
        return self.recv_data(mysocket)

    # This method handles the data that we receive and checks it to determine what our crawler needs to do with the
    # links it is being given.
    def recv_data(self, mysocket):
        global sessionid
        global cookie
        mysocket.settimeout(.3)
        data_list = []

        # This try-except ensures that we receive the data we need and times out if the server is taking too long to
        # respond.
        try:
            data = mysocket.recv(10000000).decode('ascii')
            data_list = data.split("\r\n\r\n\n\n\n")
        except timeout:
            mysocket.shutdown(1)
            mysocket.close()
        # Handles how we receive data when we are crawling through the website
        if crawling:
            if len(data_list) == 1:
                data_dict = self.handle_http(data_list[0])
                if "503" in data_list[0]:
                    self.send_get_message(frontier[1], cookie)
                    return
                if "403" in data_list[0] or "404" in data_list[0]:
                    frontier.remove(frontier[1])
                    return
            if len(data_list) == 2:
                self.handle_html(data_list[1])
                return data_list[1]

        # These else-if statements handle the data for when we are both logged into the site and not logged in.
        elif len(data_list) == 1 and not logged_in:
            data_dict = self.handle_http(data_list[0])
            return self.send_get_message(data_dict['Location'], "")
        elif len(data_list) == 1 and logged_in:
            data_dict = self.handle_http(data_list[0])
            cookie = "Cookie: csrftoken=" + data_dict['Set-Cookie'] + "; sessionid=" + data_dict['sessionid']
            sessionid = data_dict['sessionid']
            return self.send_get_message(data_dict['Location'], cookie)
        elif len(data_list) == 2:
            http_dict = self.handle_http(data_list[0])
            self.handle_html(data_list[1])
            return http_dict

    # Handles the html part of the response
    @staticmethod
    def handle_html(data):
        html_parser = MyHTMLParser()
        html_parser.feed(data)

    # Handles the http part of the response
    @staticmethod
    def handle_http(data):
        response_dict = {}
        split_response = data.splitlines()
        # Splits the response by line and turns it into a dictionary
        for line in split_response:
            if ":" in line:
                parts = line.split(":")
                if parts[0] not in response_dict:
                    response_dict[parts[0]] = parts[1].strip()
                else:
                    response_dict['sessionid'] = parts[1].strip().split(";")[0].replace("sessionid=", "")
        if not logged_in:
            response_dict["Set-Cookie"] = response_dict['Set-Cookie'].split(";")[0].replace("csrftoken=", "")
        return response_dict

    # Logs us in to fakebook by sending the POST message
    def login(self):
        global logged_in
        global crawling
        http_dict = self.send_get_message(frontier[0], '')
        login_info = "username=%s&password=%s&csrfmiddlewaretoken=%s" % (self.username, self.password, token)
        self.session_cookie = http_dict['Set-Cookie']

        # Creates the format of the post message to send to the server in order to login
        post = "POST %s HTTP/1.1\r\nHost: %s\r\nConnection: close" % ('/accounts/login/?next=/fakebook/', self.server)
        content_type = "Content-Type: application/x-www-form-urlencoded"
        content_length = "Content-Length: " + str(len(login_info))
        cookie = "Cookie: " + "sessionid=" + http_dict['sessionid'] + "; csrftoken=" + http_dict['Set-Cookie']
        msg = post + "\r\n" + content_length + "\r\n" + content_type + "\r\n" + cookie + "\r\n\r\n" + login_info + '\r\n\r\n'
        context = ssl.create_default_context()
        mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mysocket = context.wrap_socket(mysocket, server_hostname=self.server)
        mysocket.connect((self.server, self.port))
        mysocket.send(msg.encode('ascii'))
        logged_in = True
        post_login = self.recv_data(mysocket)
        visited.append("/fakebook/")
        crawling = True
        return post_login

    # Crawls through the website until we find 5 flags.
    def crawl(self, home_page):
        global cookie
        cookie = "Cookie: " + home_page['Set-Cookie'] + "; sessionid=" + sessionid
        loop_frontier = copy.deepcopy(frontier)
        for link in loop_frontier:
            if link not in visited:
                recv = self.send_get_message(link, cookie)
                frontier.remove(link)
                visited.append(link)

        # Continues to loop through the links until we have found all the flags needed
        while len(flags) < 5:
            loop_frontier = copy.deepcopy(frontier)
            for link in loop_frontier:
                if link not in visited:
                    if len(flags) == 5:
                        return
                    frontier.remove(link)
                    recv = self.send_get_message(link, cookie)
                    visited.append(link)


# Implements the HTML parser class, so we can parse HTML responses from fakebook
class MyHTMLParser(HTMLParser):

    # Handles the start tag of an HTML response to check if it is a link or not
    def handle_starttag(self, tag, attrs):
        global token

        # Checks if we are on the "first run" of the site and looks for a specific tag
        if len(first_run) == 0 and tag == 'a':
            if attrs[0][1] == "/fakebook/":
                frontier.append(attrs[0][1])
                first_run.append('LOL')
                return
        elif len(first_run) != 0 and tag == 'a':
            domain = attrs[0][1].split('/')
            if len(domain) > 1:
                domain = domain[1]
            if tag == 'a' and domain == 'fakebook':
                if attrs[0][1] not in visited:
                    frontier.append(attrs[0][1])
            if tag == 'h2':
                flags.append(attrs)
        if tag == "input":
            yes = False
            for attr in attrs:
                if attr[1] == 'csrfmiddlewaretoken':
                    yes = True
                if yes and attr[0] == 'value':
                    token = attr[1]

    def handle_endtag(self, tag):
        pass

    # Prints the flags when found in the HTML data responses
    def handle_data(self, data):
        global flags
        if 'FLAG' in data:
            flags.append(data)
            flag = data.split(":")[1].strip()
            print(flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='crawl Fakebook')
    parser.add_argument('-s', dest="server", type=str, default=DEFAULT_SERVER, help="The server to crawl")
    parser.add_argument('-p', dest="port", type=int, default=DEFAULT_PORT, help="The port to use")
    parser.add_argument('username', type=str, help="The username to use")
    parser.add_argument('password', type=str, help="The password to use")
    args = parser.parse_args()
    sender = Crawler(args)
    sender.run()

```