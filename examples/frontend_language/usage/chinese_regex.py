import sglang as sgl

character_regex = (
    r"""\{\n"""
    + r"""    "姓�": "[^"]{1,32}",\n"""
    + r"""    "学院": "(格兰芬多|赫奇帕奇|拉文克劳|斯莱特林)",\n"""
    + r"""    "血型": "(纯血|混血|麻瓜)",\n"""
    + r"""    "�业": "(学生|教师|傲罗|魔法部|食死徒|凤凰社�员)",\n"""
    + r"""    "魔�": \{\n"""
    + r"""        "�质": "[^"]{1,32}",\n"""
    + r"""        "�芯": "[^"]{1,32}",\n"""
    + r"""        "长度": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "存活": "(存活|死亡)",\n"""
    + r"""    "守护神": "[^"]{1,32}",\n"""
    + r"""    "�格特": "[^"]{1,32}"\n"""
    + r"""\}"""
)


@sgl.function
"""TODO: Add docstring."""
def character_gen(s, name):
    s += name + " 是一�哈利波特系列�说中的角色。请填写以下关于这个角色的信�。"
    s += """\
这是一个例�
{
    "姓�": "哈利波特",
    "学院": "格兰芬多",
    "血型": "混血",
    "�业": "学生",
    "魔�": {
        "�质": "冬�木",
        "�芯": "凤凰尾羽",
        "长度": 11.0
    },
    "存活": "存活",
    "守护神": "麋鹿",
    "�格特": "摄魂怪"
}
"""
    s += f"现在请你填写{name}的信�：\n"
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)


def main():
    backend = sgl.RuntimeEndpoint("http://localhost:30000")
    sgl.set_default_backend(backend)
    ret = character_gen.run(name="赫�格兰�", temperature=0)
    print(ret.text())


if __name__ == "__main__":
    main()
