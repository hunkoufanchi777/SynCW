from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib

import requests
import random
import re


def validateEmail(email):

    if len(email) > 7:
        if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) != None:
            return 1
    return 0


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def get_sentence():
    # 获取金山词霸每日一句，英文和翻译
    url = "http://open.iciba.com/dsapi/"
    r = requests.get(url)
    # print(r.json())
    contents = r.json()['content']
    note = r.json()['note']
    translation = r.json()['translation']
    print(contents, note)
    return contents + '\n' + note


def get_words():
    res = ['Wubba lubba dub dub', 'haha']
    # try:
    #     con = requests.get('http://api.eei8.cn/say/api.php?encode=js')
    #     # print(con.content.decode('utf-8'))
    #     res = re.findall("document.write\('(.*?)'\);", con.content.decode('utf-8'))
    #     print(res[0])
    #
    #     # con = requests.get('https://v1.hitokoto.cn')
    #     # con = requests.get('https://international.v1.hitokoto.cn')
    #     # print(con.content.decode('utf-8'))
    #     # res = re.findall("\"hitokoto\":\"'(.*?)'\";", con.content.decode('utf-8'))
    #     # print(res[0])
    #
    # except:
    #     res = ['Wubba lubba dub dub', 'haha']
    #     print(res[0])

    return res[0]


# if __name__ == '__main__':
#
#     import time
#
#     QQmail = MailLogs()
#     QQmail.sendmail('test', header=get_words())
#
#     while 1:
#         current_time = time.localtime(time.time())
#         if current_time.tm_hour == 23 and current_time.tm_min == 54 and current_time.tm_sec == 10:
#             print("test success")
#         if current_time.tm_hour == 8 and current_time.tm_min == 0 and current_time.tm_sec == 0:
#             QQmail.set_to('799802172@qq.com')
#             QQmail.sendmail(get_sentence(), header=get_words())
#             time.sleep(1)
#
#         time.sleep(1)

