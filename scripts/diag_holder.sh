#!/bin/sh
# 找出持有 /dev/video* 和 v4l-subdev2 的进程
for d in /proc/[0-9]*/fd; do
  if ls -l "$d" 2>/dev/null | grep -qE '/dev/(video[0-9]+|v4l-subdev)'; then
    pid=${d#/proc/}
    pid=${pid%/fd}
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    echo "pid=$pid cmd=$cmd"
    ls -l "$d" 2>/dev/null | grep -oE '/dev/(video[0-9]+|v4l-subdev[0-9]+)' | sort -u | sed 's/^/    holds /'
  fi
done
echo ----
ps -A 2>/dev/null | grep -iE 'camera|hdmi' | head -6
