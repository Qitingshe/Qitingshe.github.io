import time
import re
import os


def filename(name):
    date = time.strftime('%Y-%m-%d', time.localtime())
    return date+"-"+name.replace(' ', '-')


def newfile(fname):
    md = open('_posts/'+fname+".md", 'w')
    title = "\ntitle:\t\t"+input("input title:")
    subtitle = "\nsubtitle:\t"+input("input subtitle:")
    date = "\ndate:\t\t"+time.strftime('%Y-%m-%d', time.localtime())
    author = "\nauthor:\t\t"+"QITINGSHE"
    h_img = input("input header-img name(默认swift.jpg):")
    if h_img == "":
        h_img = "swift.jpg"
    header_img = "\nheader-img:\timg/post-bg-"+h_img
    tag = input("input tags:")
    tags = "\ntags:\n    - "+tag.replace(" ", "\n    - ")
    message = "---\n"+"layout:\t\tpost"+title+subtitle+date + \
        author+header_img+"\ncatalog: true"+tags+"\n---\n\n"
    print(message)
    md.write(message)
    md.close()

    print("==========================\n创建新文件："+fname+".md\n==========================")


def imagepath(name=""):
    pat = "\]\(pic/"
    if name == "":
        name = "_posts/" + input("input filename:")+".md"
    else:
        name = "_posts/"+filename(name)+".md"
    md = open(name, "r")
    f = md.read()
    pattern = r"](https://github.com/Qitingshe/Qitingshe.github.io/raw/master/_posts/assets/"
    pat = "\]\(../_posts/assets/"
    f1 = re.sub(pat, pattern, f)
    md.close()
    md = open(name, "w")
    md.write(f1)
    md.close()
    print("==========================\n已经将所有图片路径替换完毕\n==========================")


if __name__ == "__main__":
    name = input("input name:")
    while(True):
        print('1:新创建文件，2：修改文章中图片路径,其他按键退出')
        num = input("input 1(新建博客) or 2(修改图片路径) or other_key:")
        if num == '1':
            fname = filename(name)
            newfile(fname)
            os.system("/opt/Typora/Typora _posts/"+fname+".md")
        elif num == '2':
            print(name == "")
            imagepath(name)
        else:
            print("退出")
            break
