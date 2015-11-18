#ifndef LAYERS_CTCPOSTOUTPUTLAYER_HPP
#define LAYERS_CTCPOSTOUTPUTLAYER_HPP

#include "PostOutputLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * This layer is used as the very last layer which store the target outputs and calculates the
     * error between the target outputs and the actual outputs
     *
     * @param TDevice The computation device (Cpu or Gpu)
     * *********************************************************************************************/
    template <typename TDevice>
    class CtcPostOutputLayer : public PostOutputLayer < TDevice >
    {
        typedef typename TDevice::real_vector real_vector;
        typedef typename TDevice::int_vector int_vector;

    private:          
    
        const int         m_prevLayerSize;
        size_t            m_maxWordLen;
        size_t            m_computationSize;
    
        int_vector        m_imap;
        int_vector        m_fmap;
        int_vector        m_gmap;
        
        real_vector       m_betas;
        real_vector       m_alphas;
     
        real_vector       m_setBeta;
        real_vector       m_setAlpha;       
        real_vector       m_wordErrors;

        /**
         * Forward pass computation
         */
        void findAlphas();
        
        /**
         * Backward pass computation
         */
        void findBetas();
    public:
        /**
        * Constructs the Layer
        *
        * @param layerChild     The layer child of the JSON configuration for this layer
        * @param precedingLayer The layer preceding this one
        */
        CtcPostOutputLayer(
            const helpers::JsonValue &layerChild,
            Layer<TDevice> &precedingLater
            );

        /**
         * Destructs the Layer
         */
        virtual ~CtcPostOutputLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see PostOutputLayer::calculateError()
         */
        virtual real_t calculateError();

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

        /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();

        /**
         * @see Layer::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);
    };
} // namespace layers


#endif // LAYERS_CTCPOSTOUTPUTLAYER_HPP
