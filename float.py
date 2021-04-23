if __name__=="__main__":

    user = input ("Enter input value")
    try:
        if "." in user :
            val = float(user)
            return True
        else:
            val = int(user)
            return True
    except ValueError:
        return False