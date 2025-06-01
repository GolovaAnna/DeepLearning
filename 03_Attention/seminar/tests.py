from Summarizer import Summarizer
from transformer import EncoderDecoder
from attention_map import attention_map

#1

# #2
# model = EncoderDecoder(word_field)
# model.load_state_dict(torch.load('model.pt'))
# model.eval()

# summarizer = Summarizer(model)
# input_text = "В международном аэропорту Гонконга автомобиль технической службы столкнулся с самолетом, на борту которого находились 295 человек. В результате пострадал водитель машины. Видео инцидента в четверг, 8 сентября, публикует издание The Daily Mail. Инцидент произошел в момент, когда лайнер ехал к взлетной полосе, чтобы направиться в малайзийский штат Пинанг. Фургон врезался в двигатель воздушного судна. В результате водитель получил травмы плеча и головы. Чтобы вызволить его из покореженного автомобиля, понадобились усилия пяти человек."
# attention_map(summarizer, input_text, 'decoder_self')

def test_predict(model, input_text):
    model.eval()
    summarizer = Summarizer(model)
    return summarizer.predict(input_text)

#  model.load_state_dict(torch.load('model.pt'))
# input_text = "В международном аэропорту Гонконга автомобиль технической службы столкнулся с самолетом, на борту которого находились 295 человек. В результате пострадал водитель машины. Видео инцидента в четверг, 8 сентября, публикует издание The Daily Mail. Инцидент произошел в момент, когда лайнер ехал к взлетной полосе, чтобы направиться в малайзийский штат Пинанг. Фургон врезался в двигатель воздушного судна. В результате водитель получил травмы плеча и головы. Чтобы вызволить его из покореженного автомобиля, понадобились усилия пяти человек."
def test_attention_map(model, input_text, type='decoder_self'):
    summarizer = Summarizer(model)
    attention_map(summarizer, input_text, type)