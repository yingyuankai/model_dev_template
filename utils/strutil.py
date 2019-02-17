#!/usr/bin/env python
# -*-coding:utf-8-*-

import os
import re
import types
import curses.ascii
import random

# find surround string
SENTENCE_DIVIDER = [',', '，', "？", "。", "：", "！", "；", " ", "\""]
SENTENCE_END = ['。', '?', '？', '!', '！', '\n', ';', '；']
STR_UTIL_STOP_CHARS = set(
    [u',', u' ', u'"', u'`', u'~', u'·', u'!', u'！', u'@', u'#', u'$', u'%', u'%', u'^', u'…', u'&', u'*', u'(', u')',
     u'（', u'）', u'-', u'_', u'+', u'=', u'{', u'}', u'[', u']', u'|', u'\\', u':', u'：', u';', u'；', u'"', u'”', u'“',
     u'\'', u'<', u'《', u'》', u'>', u'，', u',', u'?', u'？', u'‘', u'’', u'/', u'、', u'。', u'.', u'￥', u'～', u'＂', u'　'])
SENTENCE_END2 = ['。', '？', "！"]


def parseLtpBody(body):
    '''
        解析parser结果
        return [{word:xx,  pos:xx, parent:xx, relate:xx, word_id:xx}]
    '''
    info_list = []
    tmp_list = body.strip().split(" ")
    for word_str in tmp_list:
        tmp1_list = word_str.strip().rsplit("/")
        if len(tmp1_list) != 5:
            return []
        word, pos, word_id, parent_id, relate = tmp1_list
        info_list.append(
            {"word": word, "pos": pos, "parent": int(parent_id), "relate": relate, "word_id": int(word_id)})
    return info_list


def findSurround(bPri, pos, pattern_len, content_uni):
    content_len = len(content_uni)
    if bPri:
        start = pos + pattern_len
        end = content_len
        inc = 1
        if start >= end:
            return None
    else:
        start = pos - 1
        end = -1
        inc = -1
        if start <= end:
            return None
    index = start
    while index != end and index >= 0 and index < content_len:
        check_char = content_uni[index].encode("utf-8")
        if check_char in SENTENCE_DIVIDER:
            index -= inc
            break
        index += inc
    if index < 0:
        index = 0
    if index >= content_len:
        index = content_len - 1
    if (bPri and index < start) or ((not bPri) and start < index):
        return None

    if start < index:
        return content_uni[start:index + 1].encode("utf-8")
    else:
        return content_uni[index:start + 1].encode("utf-8")


def splitFiles(incTxt, line_len, divCnt):
    if line_len % divCnt != 0:
        lineCnt = line_len / divCnt + 1
    else:
        lineCnt = line_len / divCnt
    os.system("split -d -l %d %s" % (lineCnt, os.path.basename(incTxt)))


def splitString(line, split_end=SENTENCE_END):  # 单字符切分句子，保留切分符号
    data = []
    line_uni = line
    if isinstance(line, types.UnicodeType) != True:
        line_uni = line.decode("utf-8")
    idx = 0
    start = 0
    splitToken = set(split_end)
    for item in line_uni:
        item_u = item.encode("utf-8")
        if item_u in splitToken:
            data.append(line_uni[start:idx + 1])
            start = idx + 1
        idx += 1
    if start < idx:
        data.append(line_uni[start:])
    return data


def splitSentenceByParser(info_list, sentence_end=SENTENCE_END):
    '''
        切句子, 返回[sen1, sen2, ...], 其中sen的格式[{word:xx,  pos:xx, parent:xx, relate:xx, word_id:xx}, {}, ..]
    '''
    sen_list = []
    sub_sen = []
    for info in info_list:
        sub_sen.append(info)
        word = info.get("word", "")
        if word in sentence_end:
            if len(sub_sen) > 1:
                sen_list.append(sub_sen)
                sub_sen = []
    if len(sub_sen) > 1:
        sen_list.append(sub_sen)
    return sen_list


# 根据句号问号等切句子
def splitSentence(content, sen_end=set(SENTENCE_END), object_end=False):
    uni_content = content
    sen_end = set([char for char in sen_end])
    # sen_end = set([u'。',u'？',u'！',u'?',u'!',u"；",u';'])
    tmp_str = ""
    ret_list = []
    for char in uni_content:
        if char in sen_end:
            if tmp_str != "":
                if object_end:
                    tmp_str += char
                ret_list.append(tmp_str)
            tmp_str = ""
        else:
            tmp_str += char
    if tmp_str != "":
        ret_list.append(tmp_str)
    return ret_list


# char basic function
def isAlphaOrNum(string):
    for c in string:
        if (not c.isalpha()) and (not c.isdigit()):
            return False
    return True


def hasNum(string):
    for c in string:
        if (c.isdigit()):
            return True
    return False


def hasBlank(string):
    for c in string:
        if (curses.ascii.isblank(c)):
            return True
    return False


def isFullAlphaOrNum(string):
    for c in string:
        if (not c.isalpha()) and (not c.isdigit()):
            return False
    return True


def isFullDigit(string):
    for c in string:
        if (not c.isdigit()):
            return False
    return True


def isSingleAlpha(string):
    return (len(string) == 1 and string[0].isalpha())


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def removeUChar(string, re_char):
    ret = "".decode("utf-8")
    for char in string:
        if char != re_char:
            ret += char
    return ret


def removeUChars(string, re_char):
    ret = "".decode("utf-8")
    for char in string:
        if not char in re_char:
            ret += char
    return ret


def replaceConChar(string):
    newln = ""
    for ch in string:
        if not curses.ascii.iscntrl(ch):
            newln += ch
    return newln


def replaceNonWord(string):
    newln = ""
    for ch in string:
        if curses.ascii.iscntrl(ch):
            continue
        if curses.ascii.isblank(ch):
            continue
        newln += ch
    return newln


def tobwords(text, tolower=True, exstop=False, tagpos=False):
    '''
    exstop: is exclude stop flag word
    tagpos: is tag the word pos
    '''
    if not text:
        return []
    if type(text) == str:
        text = text.decode("utf8")
    if tolower:
        text = text.lower()
    text = text.lower()
    words = []
    status = 0
    bgram = ""
    index = 0
    for c in text:
        index += 1
        if is_chinese(c):
            if status > 2:
                if not tagpos:
                    words.append(bgram)
                else:
                    words.append((bgram, status))
                status = 2
            elif status >= 1:
                if not tagpos:
                    words.append(bgram + c)
                else:
                    words.append((bgram + c, status))
                status = 1
            else:
                status = 2
            bgram = c
        elif c >= '0' and c <= '9':
            if status == 3:
                bgram += c
            else:
                if status >= 2:
                    if not tagpos:
                        words.append(bgram)
                    else:
                        words.append((bgram, status))
                bgram = c
            status = 3
        elif c == '.' and status == 3 and index < len(text) and text[index] >= '0' and text[index] <= '9':
            bgram += c
        elif (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'):
            if status == 4:
                bgram += c
            else:
                if status >= 2:
                    if not tagpos:
                        words.append(bgram)
                    else:
                        words.append((bgram, status))
                bgram = c
            status = 4
        else:
            if status >= 2:
                if not tagpos:
                    words.append(bgram)
                else:
                    words.append((bgram, status))
            if not exstop or c not in STR_UTIL_STOP_CHARS:
                if not tagpos:
                    words.append(c)
                else:
                    if c in STR_UTIL_STOP_CHARS:
                        words.append((c, 0))
                    else:
                        words.append((c, 5))
            bgram = ""
            status = 0
    if status >= 2:
        if not tagpos:
            words.append(bgram)
        else:
            words.append((bgram, status))
    return words


def touwords(text, tolower=True, exstop=False, tagpos=False):
    '''
    exstop: is exclude stop flag word
    tagpos: is tag the word pos
    '''
    if not text:
        return []
    if type(text) == str:
        text = text.decode("utf8")
    if tolower:
        text = text.lower()
    text = text.lower()
    words = []
    status = 0
    bgram = ""
    index = 0
    for c in text:
        index += 1
        if is_chinese(c):
            if status > 2:
                if not tagpos:
                    words.append(bgram)
                else:
                    words.append((bgram, status))
            status = 2
            if not tagpos:
                words.append(c)
            else:
                words.append((c, status))
            bgram = ""
        elif c >= '0' and c <= '9':
            if status == 3:
                bgram += c
            else:
                if status > 3:
                    if not tagpos:
                        words.append(bgram)
                    else:
                        words.append((bgram, status))
                bgram = c
            status = 3
        elif c == '.' and status == 3 and index < len(text) and text[index] >= '0' and text[index] <= '9':
            bgram += c
        elif (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'):
            if status == 4:
                bgram += c
            else:
                if status >= 3:
                    if not tagpos:
                        words.append(bgram)
                    else:
                        words.append((bgram, status))
                bgram = c
            status = 4
        else:
            if status >= 3:
                if not tagpos:
                    words.append(bgram)
                else:
                    words.append((bgram, status))
            if not exstop or c not in STR_UTIL_STOP_CHARS:
                if not tagpos:
                    words.append(c)
                else:
                    if c in STR_UTIL_STOP_CHARS:
                        words.append((c, 0))
                    else:
                        words.append((c, 5))
            bgram = ""
            status = 0
    if status > 2:
        if not tagpos:
            words.append(bgram)
        else:
            words.append((bgram, status))
    return words


def getRandomStr(slen=8, charsets=None):
    if charsets == None or charsets == "":
        charsets = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIGKLMNOPQRSTUVWXYZ";
    return "".join(random.sample(charsets, slen))


def getResultForDigit(a, encoding="utf-8"):
    dict = {u'零': 0, u'一': 1, u'二': 2, u'三': 3, u'四': 4, u'五': 5, u'六': 6, u'七': 7, u'八': 8, u'九': 9, u'十': 10,
            u'百': 100, u'千': 1000, u'万': 10000, u'０': 0, u'１': 1, u'２': 2, u'３': 3, u'４': 4, u'５': 5, u'６': 6, u'７': 7,
            u'８': 8, u'９': 9, u'壹': 1, u'贰': 2, u'叁': 3, u'肆': 4, u'伍': 5, u'陆': 6, u'柒': 7, u'捌': 8, u'玖': 9, u'拾': 10,
            u'佰': 100, u'仟': 1000, u'萬': 10000, u'亿': 100000000}
    if isinstance(a, str):
        a = a.decode(encoding)
    count = 0
    result = 0
    tmp = 0
    Billion = 0
    dpoint = 0
    while count < len(a):
        tmpChr = a[count]
        # print tmpChr
        if tmpChr.isdigit() == False:
            tmpNum = dict.get(tmpChr, None)
        else:
            tmpNum = int(tmpChr)
        # 如果等于1亿
        if tmpNum == 100000000:
            result = result + tmp
            result = result * tmpNum
            # 获得亿以上的数量，将其保存在中间变量Billion中并清空result
            Billion = Billion * 100000000 + result
            result = 0
            tmp = 0
        # 如果等于1万
        elif tmpNum == 10000:
            result = result + tmp
            result = result * tmpNum
            tmp = 0
        # 如果等于十或者百，千
        elif tmpNum >= 10 and dpoint == 0:
            if tmp == 0:
                tmp = 1
            result = result + tmpNum * tmp
            tmp = 0
        # 如果是个位数
        elif tmpNum is not None and dpoint == 0:
            tmp = tmp * 10 + tmpNum
        # 小数点处理
        elif (tmpNum is None and tmpChr.encode('utf-8') == '.') or dpoint != 0:
            # print tmp,tmpNum,dpoint,result
            if dpoint == 0:
                dpoint = 1
            else:
                tmp = tmp + tmpNum * 0.1 ** dpoint
                dpoint += 1
        count += 1
    result = result + tmp
    result = result + Billion
    return result


def rmNoncharUnicode(line):
    new_line = "".decode("utf-8")
    for uchar in line:
        if uchar >= u'\uE000' and uchar <= u'\uE0FF':
            continue
        new_line += uchar
    return new_line


def strQ2B(ustring):
    """把字符串全角转半角"""
    rstring = ""
    if type(ustring) != type(u''):
        ustring = ustring.decode("utf-8")
    for uchar in ustring:
        inside_code = ord(uchar)
        # 全角区间
        if inside_code >= 0xFF01 and inside_code <= 0xFF5E:
            inside_code -= 0xfee0
            rstring += inside_code
        # 全角空格特殊处理
        elif inside_code == 0x3000:
            inside_code = 0x0020
            rstring += inside_code
        else:
            rstring += uchar
    return rstring.encode("utf-8")


def strB2Q(ustring):
    """把字符串半角转全角"""
    rstring = ""
    if type(ustring) != type(u''):
        ustring = ustring.decode("utf-8")
    for uchar in ustring:
        inside_code = ord(uchar)
        # 半角区间
        if inside_code >= 0x0021 and inside_code <= 0x7e:
            inside_code += 0xfee0
            rstring += inside_code
        # 全角空格特殊处理
        elif inside_code == 0x0020:
            inside_code = 0x3000
            rstring += inside_code
        else:
            rstring += uchar
    return rstring.encode("utf-8")


def is_num(string):
    """ 判断是否是一个数字 """
    if string == "":
        return False, False
    bag_char = set(['.', '%', "-", ","])
    if string in bag_char:
        return False, False
    num_char = set([u'.', u',', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'.', u'％', u'%', u'-'])
    raw_num = set([u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9'])
    # uni_string = string.decode("utf-8")
    uni_string = string
    ret_str = ""
    has_raw = True
    for uch in uni_string:
        if uch not in num_char:
            return False, False
        if uch not in raw_num:
            has_raw = False
    return True, has_raw


def has_nochinese(string):
    uni_str = string.decode("utf-8")
    for char in uni_str:
        if is_chinese(char):
            return False
    return True


def refine_text(words, stopwords):
    """
    refine text:
    1) delete stopwrods;
    2) num to <num>
    :param words:
    :return:
    """
    new_words = []
    pos = []
    for word, flag in words:
        if word in stopwords or word =='\r\n':
            continue
        # if is_num(word)[0]:
        #     word = '<num>'
        if word == '×':
            word = '<thing>'
        if flag == 'nr':
            word = '<people>'
        new_words.append(word)
        pos.append(flag)
    return new_words, pos


if __name__ == '__main__':
    # splitFiles("person.log",122261,10)
    text = u'这國是1❷❷❷ｈｈｈｈｈ中古个chinese2.5的测试,包括空格 、？￥$最好的测试①②③四五⑥78.25%〇1'
    for c in text:
        print(c, is_chinese(c))
    words = touwords(text, exstop=True, tagpos=True)
    for (word, pos) in words:
        print(word, pos)
    text = u'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ【】！（）｛｝”“'
    print(strQ2B(text))
    text = "我爱北京天安门同花顺。我在那里!123"
    print("|".join(splitSentence(text)))
