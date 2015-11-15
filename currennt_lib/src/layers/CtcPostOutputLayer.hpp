#ifndef LAYERS_CTCPOSTOUTPUTLAYER_HPP
#define LAYERS_CTCPOSTOUTPUTLAYER_HPP
#include "PostOutputLayer.hpp"
namespace layers {
    template <typename TDevice>
    class CtcPostOutputLayer : public PostOutputLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;
        typedef typename TDevice::int_vector int_vector;

        public:
            CtcPostOutputLayer(
                    const helpers::JsonValue &layerChild, 
                    Layer<TDevice> &precedingLayer
                    );
            virtual ~CtcPostOutputLayer();
            virtual const std::string& type() const;
            virtual real_t calculateError();
            virtual void computeForwardPass();
            virtual void computeBackwardPass();
        
            void loadSequences(const data_sets::DataSetFraction &fraction);

        private:
            void findAlphas();
            void findBetas();

            int_vector imap;
            int_vector fmap;
            int_vector gmap;
           
            real_vector m_betas;
            real_vector m_alphas;

            real_vector wordErrors;

            size_t maxWordLen;
            size_t computationSize;
            int prevLayerSize;

            real_vector setBeta;
            real_vector setAlpha;
    };
} // namespace layers
#endif // LAYERS_CTCPOSTOUTPUTLAYER_HPP
