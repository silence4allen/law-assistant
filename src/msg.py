# -*- coding: utf-8 -*-#
class Msg:
    def __init__(self, role, content, reply_text="", think_text="", reference_nodes=None):
        self.role = role
        self.content = content
        self.reply_text = reply_text
        self.think_text = think_text
        self.reference_nodes = reference_nodes
