#pragma once

#include <av/Frame.hpp>
#include <av/common.hpp>

namespace av
{

class Resample : NoCopyable
{
	explicit Resample(SwrContext* swr) noexcept
	    : swr_(swr)
	{}

public:
	static Expected<Ptr<Resample>> create(int inChannels, AVSampleFormat inSampleFmt, int inSampleRate,
                                      int outChannels, AVSampleFormat outSampleFmt, int outSampleRate) noexcept
	{
		try {
			SwrContext* swr = nullptr;
			
			// 使用新的 FFmpeg API (>= 6.0)
			AVChannelLayout inLayout = {}, outLayout = {};
			
			// 设置默认声道布局 - 替换 av_get_default_channel_layout
			av_channel_layout_default(&inLayout, inChannels);
			av_channel_layout_default(&outLayout, outChannels);
			
			LOG_AV_DEBUG("Creating swresample context: {} {} {} -> {} {} {}",
						av_get_sample_fmt_name(inSampleFmt), inSampleRate, inChannels,
						av_get_sample_fmt_name(outSampleFmt), outSampleRate, outChannels);
			
			// 使用新的 swr_alloc_set_opts2 API - 替换 swr_alloc_set_opts
			int ret = swr_alloc_set_opts2(&swr,
										&outLayout, outSampleFmt, outSampleRate,
										&inLayout, inSampleFmt, inSampleRate,
										0, nullptr);
			
			// 清理临时布局
			av_channel_layout_uninit(&inLayout);
			av_channel_layout_uninit(&outLayout);
			
			if (ret < 0 || !swr) {
				// 使用 av::Err 的正确构造方式
				RETURN_AV_ERROR("Failed to create swr context");
			}
			
			ret = swr_init(swr);
			if (ret < 0) {
				swr_free(&swr);
				RETURN_AV_ERROR("Could not open resample context: {}", avErrorStr(ret));
			}
			
			// 修复构造函数调用 - 使用圆括号而不是花括号
			return Ptr<Resample>(new Resample(swr));
			
		} catch (const std::exception& e) {
			RETURN_AV_ERROR("Failed to create swr context");
		}
	}

	// static Expected<Ptr<Resample>> create(int inChannels, AVSampleFormat inSampleFmt, int inSampleRate,
	//                                       int outChannels, AVSampleFormat outSampleFmt, int outSampleRate) noexcept
	// {
	// 	/*
    //       * Create a resampler context for the conversion.
    //       * Set the conversion parameters.
    //       * Default channel layouts based on the number of channels
    //       * are assumed for simplicity (they are sometimes not detected
    //       * properly by the demuxer and/or decoder).
    //       */

	// 	LOG_AV_DEBUG("Creating swr context: input - channel_layout: {} sample_rate: {} format: {} output - channel_layout: {} sample_rate: {} format: {}",
	// 	             av_get_default_channel_layout(inChannels), inSampleRate, av_get_sample_fmt_name(inSampleFmt),
	// 	             av_get_default_channel_layout(outChannels), outSampleRate, av_get_sample_fmt_name(outSampleFmt));

	// 	auto swr = swr_alloc_set_opts(nullptr,
	// 	                              av_get_default_channel_layout(outChannels),
	// 	                              outSampleFmt,
	// 	                              outSampleRate,
	// 	                              av_get_default_channel_layout(inChannels),
	// 	                              inSampleFmt,
	// 	                              inSampleRate,
	// 	                              0, nullptr);

	// 	if (!swr)
	// 		RETURN_AV_ERROR("Failed to create swr context");

	// 	/* Open the resampler with the specified parameters. */
	// 	int err = 0;
	// 	if ((err = swr_init(swr)) < 0)
	// 	{
	// 		swr_free(&swr);
	// 		RETURN_AV_ERROR("Could not open resample context: {}", avErrorStr(err));
	// 	}

	// 	return Ptr<Resample>{new Resample{swr}};
	// }

	~Resample()
	{
		if (swr_)
			swr_free(&swr_);
	}

	Expected<void> convert(const Frame& input, Frame& output) noexcept
	{
		//LOG_AV_DEBUG("input - channel_layout: {} sample_rate: {} format: {}", input->channel_layout, input->sample_rate, av_get_sample_fmt_name((AVSampleFormat)input->format));
		//LOG_AV_DEBUG("output - channel_layout: {} sample_rate: {} format: {}", output->channel_layout, output->sample_rate, av_get_sample_fmt_name((AVSampleFormat)output->format));
		/* Convert the samples using the resampler. */
		auto err = swr_convert_frame(swr_, *output, *input);
		if (err < 0)
			RETURN_AV_ERROR("Could not convert input samples: {}", avErrorStr(err));

		return {};
	}

private:
	SwrContext* swr_{nullptr};
};

}// namespace av
