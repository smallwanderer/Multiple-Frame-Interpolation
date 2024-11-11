def solution(video_len, pos, op_start, op_end, commands):
    def str_to_time(s):
        a = list(map(int, s.split(':')))
        print(a)
        return 60 * a[0] + a[1]

    def move(action):
        if action == 'next':
            return 10
        else:
            return -10

    print(pos)
    pos = str_to_time(pos)
    print(pos)
    for command in commands:
        pos += move(command)
        if str_to_time(op_start) <= pos <= str_to_time(op_end):
            pos = str_to_time(op_end)
        if pos > str_to_time(video_len):
            pos = str_to_time(video_len)
        if pos < 0:
            pos = 0

    answer = ''
    return pos

print(solution("34:33","13:00","00:55","02:55",["next", "prev"]))