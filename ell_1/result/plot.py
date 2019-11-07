import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys

def parse_file(file_name):
    with open(file_name) as dataFile:
        iter_loss = dataFile.readline()
        iter_time = dataFile.readline()
    iter_loss = iter_loss.split(', ')
    iter_time = iter_time.split(', ')
    iter_loss = iter_loss[:-1]
    iter_time = iter_time[:-1]
    iter_loss = np.array(list(map(float, iter_loss)))
    iter_time = np.array(list(map(float, iter_time)))
    return iter_loss, iter_time


def trim_by_time(time, loss, time_trim):
    if time_trim < 0:
        return time, loss
    if np.max(time) < time_trim:
        return time, loss

    idx = np.argmax(time > time_trim)
    time = time[:idx]
    loss = loss[:idx]
    return time, loss


def main():
    opt = 0
    vs_time = True
    time_trim = 10
    title = ''

    if sys.argv[1] == 'rcv':
        opt = 0.168277778212676
        time_trim = 50
        title = 'rcv1'
        sys.argv[1] = 'rcv10'
    elif sys.argv[1] == 'rcv1':
        opt = 0.112102448053544
        time_trim = 50
        title = 'rcv1, mu=1'
    elif sys.argv[1] == 'rcv10':
        opt = 0.168277778212676
        time_trim = 50
        title = 'rcv1, mu=10'
    elif sys.argv[1] == 'rcv100':
        opt = 0.327721901685293
        time_trim = 50
        title = 'rcv1, mu=100'
    elif sys.argv[1] == 'news':
        opt = 0.285875660403154
        time_trim = 800
        title = 'news20.binary'
        sys.argv[1] = 'news10'
    elif sys.argv[1] == 'news1':
        opt = 0.221111221375649
        time_trim = 800
        title = 'news20.binary, mu=1'
    elif sys.argv[1] == 'news10':
        opt = 0.285875660403154
        time_trim = 800
        title = 'news20.binary, mu=10'
    elif sys.argv[1] == 'news100':
        opt = 0.421892750046794
        time_trim = 800
        title = 'news20.binary,, mu=100'
    elif sys.argv[1] == 'duke':
        opt = 0.499446323868131
        time_trim = 2.5
        title = 'duke breast-cancer'
    elif sys.argv[1] == 'mnist09':
        opt = 0.148886809671644
        time_trim = 350
        title = 'MNIST.RB 0 VS 9'
    elif sys.argv[1] == 'ijcnn1':
        opt = 0.128224776513448
        time_trim = 350
        title = 'IJCNN.RB'
    elif sys.argv[1] == 'rna':
        opt = 0.200832012210955
        time_trim = 150
        title = 'cod-rna.RB'
    else:
        print('ERROR FOLDER!')
        return

    from matplotlib.pyplot import figure
    figure(num=None, figsize=(8, 7), dpi=100, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 15})

    if sys.argv[2] == 'all':
        iter_loss1, iter_time1 = parse_file(sys.argv[1] + '/' + 'FW')
        iter_loss2, iter_time2 = parse_file(sys.argv[1] + '/' + 'APG')
        iter_loss3, iter_time3 = parse_file(sys.argv[1] + '/' + 'STORC')
        iter_loss4, iter_time4 = parse_file(sys.argv[1] + '/' + 'SCGS')
        iter_loss5, iter_time5 = parse_file(sys.argv[1] + '/' + 'SVRG')
        iter_loss6, iter_time6 = parse_file(sys.argv[1] + '/' + 'pdFW')
        loss1 = iter_loss1 - opt
        loss2 = iter_loss2 - opt
        loss3 = iter_loss3 - opt
        loss4 = iter_loss4 - opt
        loss5 = iter_loss5 - opt
        loss6 = iter_loss6 - opt
        if vs_time:
            time1 = np.cumsum(iter_time1)
            time2 = np.cumsum(iter_time2)
            time3 = np.cumsum(iter_time3)
            time4 = np.cumsum(iter_time4)
            time5 = np.cumsum(iter_time5)
            time6 = np.cumsum(iter_time6)
            time1, loss1 = trim_by_time(time1, loss1, time_trim)
            time2, loss2 = trim_by_time(time2, loss2, time_trim)
            time3, loss3 = trim_by_time(time3, loss3, time_trim)
            time4, loss4 = trim_by_time(time4, loss4, time_trim)
            time5, loss5 = trim_by_time(time5, loss5, time_trim)
            time6, loss6 = trim_by_time(time6, loss6, time_trim)
            plt.semilogy(time1, loss1, 'o-', label='FW')
            plt.semilogy(time2, loss2, 'o-', label='Acc PG')
            plt.semilogy(time3, loss3, 'o-', label='STORC')
            plt.semilogy(time4, loss4, 'o-', label='SCGS')
            plt.semilogy(time5, loss5, 'o-', label='SVRG')
            plt.semilogy(time6, loss6, 'o-', label='PDBFW (ours)')
            plt.xlabel('Running Time (seconds)')
            plt.ylabel('Relative Primal Objective')
        else:
            plt.semilogy(loss1, label='FW')
            plt.semilogy(loss2, label='Acc PG')
            plt.semilogy(loss3, label='STORC')
            plt.semilogy(loss4, label='SCGS')
            plt.semilogy(loss5, label='SVRG')
            plt.semilogy(loss6, label='Primal Dual FW')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.show()
    else:
        iter_loss, iter_time = parse_file(sys.argv[1] + '/' + sys.argv[2])
        iter_time = np.cumsum(iter_time)
        iter_loss = iter_loss - min(iter_loss)
        plt.semilogy(iter_time, iter_loss)
        plt.show()

    

if __name__ == '__main__':
    main()
