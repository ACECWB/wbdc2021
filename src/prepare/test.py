from lightgbm.sklearn import LGBMClassifier

clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=7000,
            num_leaves=256,
            reg_alpha=2.99,
            reg_lambda=1.9,
            max_depth=-1,
            subsample=1,
            colsample_bytree=1,
            random_state=2021,
            metric='None'
        )

clf