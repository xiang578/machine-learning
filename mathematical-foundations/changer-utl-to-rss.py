# coding=utf-8
# author: xiang578
# email: i@xiang578
# blog: www.xiang578.com

import re

if __name__ == '__main__':
    file = open('url.txt', 'r',)
    outfile = open('rss.xml', 'w')
    texts = []
    urls = []
    for i in file:
        url = re.findall('https?://[^/]+?/', i)
        text = i.split('http')
        s = '<outline text="'
        s = s + text[0]
        s = s + '" title="'
        s = s + text[0]
        s = s + '" type="rss" xmlUrl="'
        s = s + url[0] + 'rss/"'
        s = s + ' htmlUrl="'
        s = s + url[0]
        s = s + '"/>\n'
        outfile.write(s)
    file.close()
