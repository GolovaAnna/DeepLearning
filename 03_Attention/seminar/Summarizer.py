import torch

from preprocessing import make_mask

DEVICE = torch.device('cuda')

class Summarizer:
    """
    Генерирует заголовок по одному тексту, используя обученный EncoderDecoder и word_field.
    """
    def __init__(self, model, 
                 device=DEVICE,
                 max_len: int = 5000):
        self.model = model.to(device)
        self.word_field = self.model.word_field
        self.device = device
        self.max_len = max_len

        self.pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.bos_idx = self.word_field.vocab.stoi[self.word_field.init_token]
        self.eos_idx = self.word_field.vocab.stoi[self.word_field.eos_token]

        self.model.eval()

    def _encode(self, text: str) -> list[int]:

        tokens = self.field.preprocess(text)
        tokens = [self.field.init_token] + tokens + [self.field.eos_token]
        return [self.field.vocab.stoi.get(t,self.field.vocab.stoi[self.field.unk_token]) for t in tokens]

    def _decode(self, ids: list[int]) -> str:
        tokens = [self.field.vocab.itos[i] for i in ids]

        # Удалим служебные токены
        tokens = [t for t in tokens if t not in {self.field.init_token, self.field.eos_token}]

        return ' '.join(tokens)

    def predict(self, text: str, return_attn=False) -> str:
        """
        text : сырой текст статьи/новости
        return: сгенерированная сводка
        """
        src_tokens = self._encode(text)
        src_tensor = torch.tensor(src_tokens,
                                  dtype=torch.long,
                                  device=self.device).unsqueeze(0)


        generated = [self.bos_idx]

        for _ in range(self.max_len):
            tgt_tensor = torch.tensor(generated,
                                       dtype=torch.long,
                                       device=self.device).unsqueeze(0)

            src_mask, tgt_mask = make_mask(src_tensor, tgt_tensor, self.pad_idx)

            with torch.no_grad():
                if return_attn == False:
                    logits = self.model(src_tensor, tgt_tensor, src_mask, tgt_mask)
                else:
                    logits, attn = self.model(src_tensor, tgt_tensor, src_mask, tgt_mask, True)

                next_token = logits[0, -1].argmax(-1).item()

            if next_token == self.eos_idx or len(generated) > 20:
                break
            # if len(generated) > 20:
            #     break

            generated.append(next_token)

        # Декодируем обратно в строку (без <s>)
        if return_attn == False:
            return self._decode(generated[1:])
        else:
            return self._decode(generated[1:]), attn