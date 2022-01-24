# -*- coding: utf-8 -*-
import json
import os 

class Config(object):
    '''Class wich contains all of the configurations used'''
    #Singleton
    _instance = None

    '''Python way of utilizing singleton classes'''
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        
        #Chamada de metodo para carregar aquivo de json com as configuracoes
        self.loadData()

        self.platform               = self.data["platform"]
        self.total_frames           = self.data["total_frames"]
        self.gpu_warm_up_frames     = self.data["gpu_warm_up_frames"]
        self.different_images_sizes = self.data["different_images_sizes"]

        self.generate_img_sizes()

    def generate_img_sizes(self):
        '''Generates a tuples list of images shapes multiple of 32'''
        self.img_sizes = [(int(32*i),int(32*i),3) for i in range(3,self.different_images_sizes)]


    def loadData(self):
        '''Load the json data'''
        with open('../common/config.json') as f:
            self.data = json.load(f)["configuration"]