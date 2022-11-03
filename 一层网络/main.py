from LoR import LoR

if __name__ == '__main__':
    logi=LoR(0.0001)
    logi.data_processor()
    for i in range(500):
        logi.work()
