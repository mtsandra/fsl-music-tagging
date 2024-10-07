epoch_num = 200
input_type = "passt"
if input_type == "vggish":
    feature_size = 128 * 2
elif input_type == "passt":
    feature_size = 768 * 2
elif input_type == "openl3":
    feature_size = 512 * 2