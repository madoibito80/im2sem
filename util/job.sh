tasks=(	"shooting"
	"kuka_grasp"
	"halfcheetah")
c=0
for task in "${tasks[@]}"; do
	for t in {0..4}; do
		nohup python train_model.py --task=${task} --trial=${t} --gpu=$((t%2)) --mode=train &
		c=$((c+1))
		if [ $((c%2)) -eq 0 ];then
			sleep 30m
		fi
	done
done




