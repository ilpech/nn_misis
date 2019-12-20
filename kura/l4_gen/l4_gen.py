import matplotlib.pyplot as plt
import numpy as np
from project_tools import *
import random

MUT_R = .9
ACC = 4

def print_step(counter, popul, err_counter):
        x,y = popul[0].v2xy()
        print('pop #' + str(counter), 'val #{}: {:.5}'.format(err_counter, popul[0].cur_val), "({:.5},{:.5})".format(x,y))

class GENETICS:
    def __init__(self, size):
        self.cur_val = 10**5
        self.genom = []
        for _ in range(size):
            self.genom.append(random.randint(0,1))
        self.acc = ACC

    def v2xy(self):
        if len(self.genom) % 2 != 0:
            raise Exception
        res = []
        genom_len = len(self.genom)
        f_half = self.genom[:int(genom_len/2)-self.acc]
        s_half = self.genom[int(genom_len/2):-self.acc]
        f_float = self.genom[int(genom_len/2)-self.acc:int(genom_len/2)]
        f_f = 0
        c = 1
        for f in f_float:
            f_f += f * (2**c) * (10 ** (-c))
            c+=1
        s_float = self.genom[-self.acc:]
        s_f = 0
        c = 0
        for f in s_float:
            s_f += f * (2**c) * (10 ** (-c))
            c+=1
        x = int("".join(str(x) for x in f_half), 2)
        x = x + f_f
        y = int("".join(str(x) for x in s_half), 2) + s_f
        res.append(x)
        res.append(y)
        return res

    def mutate(self):
        should_mutate = random.random()
        if should_mutate <= MUT_R:
            gen_ind = random.randint(0, len(self.genom) - 1)
            self.genom[gen_ind] = abs(self.genom[gen_ind] - 1)

    def prisp(self):
        if abs(self.cur_err)<0.00000000000001:
            return 1
        return 1/self.cur_err

    def vprisp(self):
        if self.cur_val >= 0:
            return 1/self.cur_val
        else:
            return 1/(-self.cur_val)

def classic_crossover(o1, o2, genom_size):
    cut_point_ind = random.randint(1, len(o1.genom) - 1)
    new1_gene = o1.genom[:cut_point_ind] + o2.genom[cut_point_ind:]
    new2_gene = o2.genom[:cut_point_ind] + o1.genom[cut_point_ind:]
    new1 = GENETICS(len(o1.genom))
    new1.genom = new1_gene
    new2 = GENETICS(len(o1.genom))
    new2.genom = new2_gene
    res = (new1, new2)
    return res

def island_model_crossover(o1, o2):
    cut_point_ind1 = random.randint(0, len(o1.genom) - 1)
    new = GENETICS(len(o1.genom))
    o1_gene = o1.genom[:cut_point_ind1]
    o2_gene = o2.genom[cut_point_ind1:]
    new.genom = o1_gene + o2_gene
    return new

# FUNC = RosenbrockFunc()
FUNC = CamelFunc()
popul_size = 25
upper_border = FUNC.upper_border
genom_size = len(bin(upper_border)) - 2 
genom_size = (genom_size+ACC)*2
steps = 1000
val_arr_classic = []
val_arr_island = []

def classic(popul):
    counter = 0
    err_counter = 0
    best_val = 99999
    while err_counter < steps:
        for el in popul:
            x, y = el.v2xy()
            el.cur_val = FUNC.value(x,y)
        popul.sort(key=lambda x: x.cur_val)
        val_arr_classic.append(popul[0].cur_val)
        if counter % 100 == 0:
            print_step(counter, popul, err_counter)
        popul.sort(key=lambda x: x.vprisp())
        cur_p = 0
        pr = {}
        for el in popul:
            v = cur_p + el.vprisp()/sum([x.vprisp() for x in popul])
            pr[v] = (popul.index(el),el)
            cur_p = v
        i1 = random.randint(0, len(popul)-1)
        i2 = random.randint(0, len(popul)-1)
        while i1 == i2:
            i2 = random.randint(0, len(popul)-1)
        parents = [(i1,popul[i1]),(i2,popul[i2])]
        for i in range(2):
            r = random.random()
            for p,el in pr.items():
                if r > p:
                    parents[i] = el
        new1, new2 = classic_crossover(parents[0][1], parents[1][1], genom_size)

        n1xy = new1.v2xy()
        new1.cur_val = FUNC.value(n1xy[0], n1xy[1])
        n2xy = new2.v2xy()
        new2.cur_val = FUNC.value(n2xy[0], n2xy[1])
        popul[parents[0][0]] = new1
        popul[parents[1][0]] = new2

        popul.sort(key=lambda x: x.cur_val)

        tmp_best_val = popul[0].cur_val
        if tmp_best_val > best_val or abs(tmp_best_val - best_val) < 0.00001:
            err_counter +=1
        else:
            err_counter = 0
            print_step(counter, popul, err_counter)
            best_val = tmp_best_val
        counter += 1
    val_arr_classic.append(popul[0].cur_val)
    return counter

def island_model(popul):
    counter = 0
    err_counter = 0
    best_val = 99999
    while err_counter < steps:
        for el in popul:
            x, y = el.v2xy()
            el.cur_val = FUNC.value(x,y)
        popul.sort(key=lambda x:x.cur_val)
        val_arr_island.append(popul[0].cur_val)
        if counter % 100 == 0:
            print_step(counter, popul, err_counter)

        f_parent_ind = random.randint(0, len(popul) - 1)
        s_parent_ind = random.randint(0, len(popul) - 1)

        new_o = island_model_crossover(popul[f_parent_ind], popul[s_parent_ind])
        new_o.mutate()
        x,y = new_o.v2xy()
        new_o.cur_val = FUNC.value(x,y)

        popul[-1] = new_o

        tmp_best_val = popul[0].cur_val
        if tmp_best_val > best_val or abs(tmp_best_val - best_val) < 0.00001:
            err_counter +=1
        else:
            err_counter = 0
            print_step(counter, popul, err_counter)
            best_val = tmp_best_val
        counter += 1
    val_arr_island.append(popul[0].cur_val)
    return counter

def run(methods):
    print(methods)
    ret = []
    if "classic" in methods:
        popul = []

        for i in range(popul_size):
            popul.append(GENETICS(genom_size))
        print("CLASSIC")
        counter = classic(popul)
        print('-------')
        print('pop #' + str(counter), 'err: {:.3}'.format(popul[0].cur_val))
        print('true best value', FUNC.true_value())
        best_o_x, best_o_y = popul[0].v2xy()

        print('gene best value', FUNC.value(best_o_x, best_o_y))
        print('with x,y: ', best_o_x, best_o_y)
        ret.append('classic')
    if 'island_model' in methods:
        popul = []

        for i in range(popul_size):
            popul.append(GENETICS(genom_size))
        print()
        print("island_model")
        print()
        counter = island_model(popul)
        ret.append('island_model')
        print('-------')
        print('pop #' + str(counter), 'err: {:.3}'.format(popul[0].cur_val))
        print('true best value', FUNC.true_value())
        best_o_x, best_o_y = popul[0].v2xy()

        print('gene best value', FUNC.value(best_o_x, best_o_y))
        print('with x,y: ', best_o_x, best_o_y)
    if len(methods) == 0:
        counter = 0
        return []

    return ret

methods = ['classic', 'island_model']
pl = run(methods)
if len(pl) > 0:
    if 'classic' in pl:
        asd = 0
        qwe = []
        for i in range(len(val_arr_classic)):
            asd += val_arr_classic[i]
            qwe.append(asd/(i+1))
        plt.title(type(FUNC).__name__ + ", " + 'classic')
        plt.xlabel("Time taken")
        plt.ylabel("Val accuracy")
        plt.plot(val_arr_classic, 'y-', label='val_arr_classic')
        plt.plot(qwe, 'g-', label='mean_classic')
        plt.legend(loc='upper right')
    if len(pl) == 2:
        plt.figure()
    if 'island_model' in pl:
        asd = 0
        qwe = []
        for i in range(len(val_arr_island)):
            asd += val_arr_island[i]
            qwe.append(asd/(i+1))
        plt.title(type(FUNC).__name__ + ', ' + 'island_model')
        plt.xlabel("Time taken")
        plt.ylabel("Val accuracy")
        plt.plot(val_arr_island, 'r-', label='val_arr_island_model')
        plt.plot(qwe, 'g-', label='mean_island_model')
        plt.legend(loc='upper right')
        
plt.show()