import numpy as np
from pydub import AudioSegment, silence
import librosa
import soundfile as sf


# 01 音频处理
def extract_mfcc(segment, sr=8000, n_mfcc=10):
    """将 AudioSegment 转换为 MFCC 特征矩阵"""
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
    samples = librosa.resample(samples, orig_sr=segment.frame_rate, target_sr=sr)
    return librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc).T

def calculate_similarity(mfcc1, mfcc2):
    """计算两个 MFCC 矩阵的相似度"""
    norm1 = np.linalg.norm(mfcc1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(mfcc2, axis=1, keepdims=True)
    dot_products = np.dot(mfcc1, mfcc2.T)
    similarities = dot_products / (norm1 * norm2.T)
    return np.max(np.mean(similarities, axis=1))

def is_subset_mfcc(third_mfcc, fourth_mfcc, threshold=0.8):
    """判断第四段的 MFCC 是否是第三段的子集"""
    len_fourth = len(fourth_mfcc)
    max_similarity = 0

    for i in range(len(third_mfcc) - len_fourth + 1):
        window = third_mfcc[i:i + len_fourth]
        similarity = calculate_similarity(window, fourth_mfcc)
        max_similarity = max(max_similarity, similarity)

    return max_similarity >= threshold

def audio_to_numpy(segment):
    """将 AudioSegment 转换为 NumPy 数组"""
    return np.array(segment.get_array_of_samples(), dtype=np.int16)

def audio_post_process2(sr, audio_array, min_silence_len=200):
    """音频处理函数，找到非开头的最长停顿，只保留该停顿之前的内容"""
    audio_segment = AudioSegment(
        audio_array.tobytes(), 
        frame_rate=sr, 
        sample_width=audio_array.dtype.itemsize, 
        channels=1  # 单声道
    )

    # 静音检测
    silence_thresh = -50  # 静音阈值（dB）
    min_silence_len = min_silence_len  # 最小静音长度（毫秒）
    silences = silence.detect_silence(audio_segment, min_silence_len, silence_thresh)

    # 忽略开头的静音
    if silences and silences[0][0] <= 500:
        silences.pop(0)

    # 忽略末尾的静音
    if silences and silences[-1][1] >= len(audio_segment) - 500:
        silences.pop()

    if not silences:
        print("未检测到足够的停顿。")
        return sr, audio_array

    # 找到非开头和末尾的最长静音段
    longest_silence = max(silences, key=lambda x: x[1] - x[0])

    # 只保留该静音段之前的音频内容
    retained_segment = audio_segment[:longest_silence[0]]

    # 转换为 NumPy 数组
    retained_array = np.array(retained_segment.get_array_of_samples(), dtype=np.int16)

    # 返回采样率和保留的 NumPy 数组
    return sr, retained_array

def audio_post_process(sr, audio_array, min_silence_len=200):
    """音频处理函数，以停顿分割语音，找到相邻语音片段具有包含关系的位置，只保留该位置之前的内容"""
    audio_segment = AudioSegment(
        audio_array.tobytes(), 
        frame_rate=sr, 
        sample_width=audio_array.dtype.itemsize, 
        channels=1  # 单声道
    )

    # 静音检测
    silence_thresh = -50  # 静音阈值（dB）
    min_silence_len = min_silence_len  # 最小静音长度（毫秒）
    silences = silence.detect_silence(audio_segment, min_silence_len, silence_thresh)

    # 忽略开头的静音
    if silences and silences[0][0] <= 500:
        silences.pop(0)

    # 删除末尾的长段静音及其后面的内容
    if silences and silences[-1][1] >= len(audio_segment) - 500:
        audio_segment = audio_segment[:silences[-1][0]]
        silences.pop()

    if not silences:
        print("未检测到足够的停顿。")
        return sr, audio_to_numpy(audio_segment)

    # 提取分段音频
    segments = []
    for i in range(len(silences) + 1):
        if i == 0:
            segments.append(audio_segment[:silences[i][0]])
        elif i == len(silences):
            segments.append(audio_segment[silences[i-1][0]:])
        else:
            segments.append(audio_segment[silences[i-1][0]:silences[i][0]])

    # 提取每段音频的 MFCC 特征
    mfccs = [extract_mfcc(segment) for segment in segments]

    # 检查相邻音频段是否具有包含关系
    retain_index = len(segments)  # 默认保留所有段
    for i in range(len(mfccs) - 1):
        if is_subset_mfcc(mfccs[i], mfccs[i + 1]):
            retain_index = i + 1
            break

    # 只保留该位置之前的内容
    retained_segments = segments[:retain_index]

    # 合并音频段为 NumPy 数组
    retained_array = np.concatenate([audio_to_numpy(seg) for seg in retained_segments])

    # 返回采样率和保留的 NumPy 数组
    return sr, retained_array

# 02 静音检测与裁剪
def trim_audio_librosa(audio_data_float, top_db=30):
    """使用librosa库裁剪音频数据中的静音部分。
    
    Args:
        audio_data (np.array): 输入的音频数据数组。
        top_db (int): 静音阈值，以分贝为单位。

    Returns:
        np.array: 裁剪后的音频数据。
    """
    # librosa需要浮点数类型的音频数据
    trimmed_data, _ = librosa.effects.trim(audio_data_float, top_db=top_db)
    return trimmed_data



# 03 文本处理
punctuation = set(['!', '?', '…', ',', '.', '-'," "])
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut4(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx.append(None)
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx)-1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    opts = []
    i = 0
    while i < len(inps):
        # 尝试获取两个标点的位置
        if i + 1 < len(inps):
            # 检查两个标点切分后的字符数
            combined = "".join(inps[i:i+2])
            if len(combined) >= 35:
                # 如果字符数大于等于35，则只使用一个标点切分
                opts.append("".join(inps[i:i+1]))
                i += 1
            else:
                # 否则使用两个标点切分
                opts.append(combined)
                i += 2
        else:
            # 如果只剩一个标点或者没有标点，直接添加剩余部分
            opts.append("".join(inps[i:i+1]))
            i += 1

    # 过滤掉只包含标点的段落
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def process_long_text(text):
    # text = cut4(text)
    text = cut2(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    return texts


# 示例调用
if __name__ == "__main__":
    text = "这是一个测试文本，用于测试文本处理函数。这是第二个测试文本用于测试文本处理函数, 这是第三个测试文本用于测试文本处理函数。"
    texts = process_long_text(text)
    print(texts)
