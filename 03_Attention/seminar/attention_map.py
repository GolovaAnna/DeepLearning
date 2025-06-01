import seaborn as sns
import matplotlib.pyplot as plt

def visualize_attention(input_sentence, output_sentence, attention_weights):

    attention = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=input_sentence, yticklabels=output_sentence, cmap='viridis')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(f"Attention map {type}")
    plt.show()

def attention_map(summarizer, input_text, type):
    predict, attn = summarizer.predict(input_text, True)

    print(f'predict: {predict}')

    attn = attn[type][-1][0]
    avg_attn = attn.mean(dim=0)
    word_field = summarizer.word_field

    if type == 'encoder':

        input_tokens = word_field.preprocess(input_text)
        input_tokens = [word_field.init_token] + input_tokens + [word_field.eos_token]

        visualize_attention(
        input_sentence=input_tokens,
        output_sentence=input_tokens,
        attention_weights=avg_attn)

    if type == 'decoder_self':

        output_tokens = word_field.preprocess(predict)
        output_tokens = [word_field.init_token] + output_tokens

        visualize_attention(
        input_sentence=output_tokens,
        output_sentence=output_tokens,
        attention_weights=avg_attn)


    if type == 'decoder_enc':

        input_tokens = word_field.preprocess(input_text)
        input_tokens = [word_field.init_token] + input_tokens + [word_field.eos_token]

        output_tokens = word_field.preprocess(predict)
        output_tokens = [word_field.init_token] + output_tokens

    visualize_attention(
        input_sentence=input_tokens,
        output_sentence=output_tokens,
        attention_weights=avg_attn,  # если батч внимания
)