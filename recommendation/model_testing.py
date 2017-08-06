def model_test(algo=None, uid='1', iid='1'):
    pred = algo.predict(uid, iid)
    return pred.est
