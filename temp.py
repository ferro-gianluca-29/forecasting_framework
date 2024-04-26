def split_data(self, df):
        n = len(df)
        if self.date_list is not None:
            if self.validation:
                train = df.loc[self.date_list[0]:self.date_list[1],'date']
                valid = df[self.date_list[2]:self.date_list[3]]
                test = df[self.date_list[4]:self.date_list[5]]
                return train, test, valid
            else:
                train = df[self.date_list[0]:self.date_list[1]]
                test = df[self.date_list[2]:self.date_list[3]]
                return train, test, None 
        else:
            if  self.validation:
                train_end = int(n * self.train_size)
                valid_end = int(n * (self.train_size + self.val_size))
                                                                        
                train = df.iloc[:train_end]
                valid = df.iloc[train_end:valid_end] 
                test = df.iloc[valid_end:] 
                print(f"training: {(train.index[0],train.index[-1])} \t valid: {valid.index[0],valid.index[-1]} \t test: {test.index[0],test.index[-1]}")

                return train, test, valid
            else:
                train = df[:int(n * self.train_size)]
                test = df[int(n * self.train_size):]
                return train, test, None