
# api_key = ""
# user_name = ""
def translate(sentList, srcLang="zh", tgtLang="en"):
    import requests, json
    def genRequest(sentList, srcLang, tgtLang):
        api_key = "YOUR_API_KEY"
        user_name = "YOUR_USER_NAME"
        body = {
            "header": {
                "fn": "auto_translation",
                "api_key": api_key, # user_name & api_key can be achieved from transmart.qq.com!
                "user_name": user_name
            },
            "type": "plain",
            "model_category": "normal",
            "source": {
                "lang": srcLang,
                "text_list": sentList
            },
            "target": {
                "lang": tgtLang
            }
        }
        return body

    def translate_raw(sentList, srcLang="zh", tgtLang="en"):
        onlineUrl = "https://transmart.qq.com/api/imt"
        transBody = genRequest(sentList, srcLang, tgtLang)
        transResponse = requests.post(onlineUrl, data=json.dumps(transBody))
        transResult = json.loads(transResponse.content)['auto_translation']
        return transResult

    tryTime = 0
    result = []
    while tryTime < 10:
        try:
            result = translate_raw(sentList, srcLang, tgtLang)
            break
        except:
            tryTime += 1
    
    return result

##### demo #####


if __name__ == "__main__":
    SentList = ["It has been read for 333 times"]
    transList = translate(SentList, "en", "zh")
    print(transList)
    
    # print("===== input =====")
    # print(SentList)
    # print("===== output =====")
    # print(transList)

    # SentList = ["测试翻译系统的性能"]
    # transList = translate(SentList, "zh", "en")
    # print("===== input =====")
    # print(SentList)
    # print("===== output =====")
    # print(transList)